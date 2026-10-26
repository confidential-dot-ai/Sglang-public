//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing
//!
//! # Performance Optimizations
//! The model index uses immutable Arc snapshots instead of RwLock for lock-free reads.
//! This is critical for high-concurrency scenarios where many requests query the same model.
//!
//! # Consistent Hash Ring
//! The registry maintains a pre-computed hash ring per model for O(log n) consistent hashing.
//! The ring is rebuilt only when workers are added/removed, not per-request.
//! Uses virtual nodes (150 per worker) for even distribution and blake3 for stable hashing.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex, RwLock,
    },
};

use dashmap::DashMap;
use notify::{RecursiveMode, Watcher};
use smg_mesh::OptionalMeshSyncManager;
use uuid::Uuid;

use crate::core::model_routes::ModelRoutes;
use crate::{
    core::{
        circuit_breaker::CircuitState,
        worker::{HealthChecker, RuntimeType, WorkerType},
        ConnectionMode, Worker,
    },
    observability::metrics::Metrics,
};

/// Number of virtual nodes per physical worker for even distribution.
/// 150 is a common choice that provides good balance between memory and distribution.
const VIRTUAL_NODES_PER_WORKER: usize = 150;

/// Consistent hash ring for O(log n) worker selection.
///
/// Each worker is placed at multiple positions (virtual nodes) on the ring
/// based on hash(worker_url + vnode_index). This provides:
/// - Even key distribution across workers
/// - Minimal key redistribution when workers are added/removed (~1/N keys move)
/// - O(log n) lookup via binary search
///
/// Uses blake3 for stable, fast hashing that's consistent across Rust versions.
#[derive(Debug, Clone)]
pub struct HashRing {
    /// Sorted list of (ring_position, worker_url)
    /// Multiple entries per worker (virtual nodes) for even distribution.
    /// Uses Arc<str> to share URL across all virtual nodes (150 refs vs 150 copies).
    entries: Arc<[(u64, Arc<str>)]>,
}

impl HashRing {
    /// Build a hash ring from a list of workers.
    /// Creates VIRTUAL_NODES_PER_WORKER entries per worker for even distribution.
    pub fn new(workers: &[Arc<dyn Worker>]) -> Self {
        let mut entries: Vec<(u64, Arc<str>)> =
            Vec::with_capacity(workers.len() * VIRTUAL_NODES_PER_WORKER);

        for worker in workers {
            // Create Arc<str> once per worker, share across all virtual nodes
            let url: Arc<str> = Arc::from(worker.url());
            let url_bytes = url.as_bytes();

            // Create multiple virtual nodes per worker
            for vnode in 0..VIRTUAL_NODES_PER_WORKER {
                let mut hasher = blake3::Hasher::new();
                hasher.update(url_bytes);
                hasher.update(b"#");
                hasher.update(&(vnode as u64).to_le_bytes());
                let hash = hasher.finalize();
                let pos = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
                entries.push((pos, Arc::clone(&url)));
            }
        }

        // Sort by ring position for binary search
        entries.sort_unstable_by_key(|(pos, _)| *pos);

        Self {
            entries: Arc::from(entries.into_boxed_slice()),
        }
    }

    /// Hash a string to a ring position using blake3 (stable across versions).
    #[inline]
    fn hash_position(s: &str) -> u64 {
        let hash = blake3::hash(s.as_bytes());
        // Take first 8 bytes as u64
        u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap())
    }

    /// Find worker URL for a key using consistent hashing.
    /// Returns the first healthy worker URL at or after the key's position (clockwise).
    ///
    /// - `key`: The routing key to hash
    /// - `is_healthy`: Function to check if a worker URL is healthy
    pub fn find_healthy_url<F>(&self, key: &str, is_healthy: F) -> Option<&str>
    where
        F: Fn(&str) -> bool,
    {
        if self.entries.is_empty() {
            return None;
        }

        let key_pos = Self::hash_position(key);

        // Binary search to find first entry at or after key_pos
        let start = self.entries.partition_point(|(pos, _)| *pos < key_pos);

        // Walk clockwise from start, wrapping around
        // Track visited URLs to avoid checking same worker multiple times (virtual nodes)
        let mut checked_urls =
            std::collections::HashSet::with_capacity(self.worker_count().min(16));

        for i in 0..self.entries.len() {
            let (_, url) = &self.entries[(start + i) % self.entries.len()];
            let url_str: &str = url;

            // Skip if we already checked this worker (from another virtual node)
            if !checked_urls.insert(url_str) {
                continue;
            }

            if is_healthy(url_str) {
                return Some(url_str);
            }
        }

        None
    }

    /// Check if the ring is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the number of entries in the ring (including virtual nodes)
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Get the number of unique workers in the ring
    pub fn worker_count(&self) -> usize {
        self.entries.len() / VIRTUAL_NODES_PER_WORKER.max(1)
    }
}

/// Unique identifier for a worker
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct WorkerId(String);

impl WorkerId {
    /// Create a new worker ID
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a worker ID from a string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get the ID as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

/// Model index using immutable snapshots for lock-free reads.
/// Each model maps to an Arc'd slice of workers that can be read without locking.
/// Updates create new snapshots (copy-on-write semantics).
type ModelIndex = Arc<DashMap<String, Arc<[Arc<dyn Worker>]>>>;

#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRoutesStatus {
    pub desired_revision: Option<u64>,
    pub desired_sha256: Option<String>,
    pub accepted_revision: u64,
    pub accepted_sha256: String,
    pub error: Option<String>,
    pub watcher_error: bool,
    pub watcher_active: bool,
}

/// Worker registry with model-based indexing
#[derive(Debug)]
pub struct WorkerRegistry {
    model_routes: RwLock<ModelRoutes>,
    model_routes_file: Option<std::path::PathBuf>,
    model_routes_status: Mutex<ModelRoutesStatus>,
    worker_generation: AtomicUsize,
    alias_hash_rings: DashMap<Vec<String>, (usize, Arc<HashRing>)>,
    /// Admission fences and response counters retained through withdrawal.
    request_counts: DashMap<String, Arc<AtomicUsize>>,
    maintenance: Mutex<HashMap<String, (WorkerId, Arc<dyn Worker>)>>,
    /// All workers indexed by ID
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>,

    /// Model index for O(1) lookups using immutable snapshots.
    /// Uses Arc<[T]> instead of Arc<RwLock<Vec<T>>> for lock-free reads.
    model_index: ModelIndex,

    /// Consistent hash rings per model for O(log n) routing.
    /// Rebuilt on worker add/remove (copy-on-write).
    hash_rings: Arc<DashMap<String, Arc<HashRing>>>,

    /// Workers indexed by worker type
    type_workers: Arc<DashMap<WorkerType, Vec<WorkerId>>>,

    /// Workers indexed by connection mode
    connection_workers: Arc<DashMap<ConnectionMode, Vec<WorkerId>>>,

    /// URL to worker ID mapping
    url_to_id: Arc<DashMap<String, WorkerId>>,
    /// Optional mesh sync manager for state synchronization
    /// When None, the registry works independently without mesh synchronization
    /// Uses RwLock for thread-safe access when setting mesh_sync after initialization
    mesh_sync: Arc<RwLock<OptionalMeshSyncManager>>,
}

impl WorkerRegistry {
    /// Create a new worker registry
    pub fn new() -> Self {
        let model_routes_file =
            std::env::var_os("CI_ROUTER_MODEL_ROUTES_FILE").map(std::path::PathBuf::from);
        let (model_routes, model_routes_hash) = model_routes_file
            .as_ref()
            .map(|path| ModelRoutes::read(path).expect("invalid durable model route configuration"))
            .unwrap_or_else(|| (ModelRoutes::default(), String::new()));
        let revision = model_routes.revision;
        Self {
            model_routes_status: Mutex::new(ModelRoutesStatus {
                desired_revision: Some(revision),
                desired_sha256: Some(model_routes_hash.clone()),
                accepted_revision: revision,
                accepted_sha256: model_routes_hash,
                error: None,
                watcher_error: false,
                watcher_active: false,
            }),
            model_routes: RwLock::new(model_routes),
            model_routes_file,
            worker_generation: AtomicUsize::new(0),
            alias_hash_rings: DashMap::new(),
            request_counts: DashMap::new(),
            maintenance: Mutex::new(HashMap::new()),
            workers: Arc::new(DashMap::new()),
            model_index: Arc::new(DashMap::new()),
            hash_rings: Arc::new(DashMap::new()),
            type_workers: Arc::new(DashMap::new()),
            connection_workers: Arc::new(DashMap::new()),
            url_to_id: Arc::new(DashMap::new()),
            mesh_sync: Arc::new(RwLock::new(None)),
        }
    }

    pub fn model_routes(&self) -> ModelRoutes {
        self.model_routes.read().unwrap().clone()
    }

    pub fn resolve_model_route(&self, model: Option<&str>) -> Option<Vec<String>> {
        model.and_then(|name| self.model_routes.read().unwrap().routes.get(name).cloned())
    }

    pub fn reload_model_routes(
        &self,
        expected_revision: u64,
        expected_sha256: &str,
    ) -> Result<ModelRoutes, String> {
        let path = self
            .model_routes_file
            .as_ref()
            .ok_or("durable model routes are not configured")?;
        let (future, hash) = ModelRoutes::read(path)?;
        if hash != expected_sha256 {
            return Err("model route file hash differs".into());
        }
        // Share the withdrawal lock so the readiness check and route commit
        // cannot race with worker removal. Existing requests retain their copy.
        let _maintenance = self.maintenance.lock().unwrap();
        let mut current = self.model_routes.write().unwrap();
        if *current == future {
            let mut status = self.model_routes_status.lock().unwrap();
            if !status.accepted_sha256.is_empty() && status.accepted_sha256 != hash {
                return Err("model route bytes changed without a new revision".into());
            }
            status.error = None;
            if expected_revision == current.revision
                || current.revision.checked_sub(1) == Some(expected_revision)
            {
                return Ok(current.clone());
            }
            return Err("model route revision conflict".into());
        }
        if current.revision != expected_revision || future.revision <= current.revision {
            return Err("model route revision conflict".into());
        }
        if future.routes.values().flatten().any(|model| {
            !self.get_by_model(model).iter().any(|w| {
                w.worker_type() == &WorkerType::Regular
                    && w.connection_mode() == &ConnectionMode::Http
                    && w.is_available()
            })
        }) {
            return Err("model route has no ready regular HTTP worker".into());
        }
        // Rings are keyed by their backend list. Keep rings for unchanged
        // routes; remove only cache keys that the new document no longer uses.
        self.alias_hash_rings
            .retain(|backends, _| future.routes.values().any(|value| value == backends));
        *current = future;
        let mut status = self.model_routes_status.lock().unwrap();
        status.accepted_revision = current.revision;
        status.accepted_sha256 = hash;
        status.error = None;
        Ok(current.clone())
    }

    pub fn model_routes_status(&self) -> ModelRoutesStatus {
        self.model_routes_status.lock().unwrap().clone()
    }

    pub fn refresh_model_routes(&self) {
        let Some(path) = &self.model_routes_file else {
            return;
        };
        let desired = ModelRoutes::read(path);
        {
            let mut status = self.model_routes_status.lock().unwrap();
            match &desired {
                Ok((document, hash)) => {
                    status.desired_revision = Some(document.revision);
                    status.desired_sha256 = Some(hash.clone());
                }
                Err(error) => {
                    status.desired_revision = None;
                    status.desired_sha256 = None;
                    status.error = Some(error.clone());
                    return;
                }
            }
        }
        let (_, hash) = desired.unwrap();
        let revision = self.model_routes.read().unwrap().revision;
        if let Err(error) = self.reload_model_routes(revision, &hash) {
            self.model_routes_status.lock().unwrap().error = Some(error);
        }
    }

    pub fn watch_model_routes(self: &Arc<Self>) {
        let Some(path) = &self.model_routes_file else {
            return;
        };
        let directory = path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_owned();
        let weak = Arc::downgrade(self);
        std::thread::spawn(move || {
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            let mut watcher =
                notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                    let _ = sender.try_send(event.is_ok());
                })
                .ok();
            let mut watching = watcher.as_mut().is_some_and(|watcher| {
                watcher
                    .watch(&directory, RecursiveMode::NonRecursive)
                    .is_ok()
            });
            if let Some(registry) = weak.upgrade() {
                let mut status = registry.model_routes_status.lock().unwrap();
                status.watcher_error = !watching;
                status.watcher_active = watching;
            }
            loop {
                let event = if watching {
                    receiver.recv_timeout(std::time::Duration::from_secs(5))
                } else {
                    std::thread::sleep(std::time::Duration::from_secs(5));
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                };
                let Some(registry) = weak.upgrade() else {
                    break;
                };
                if matches!(
                    event,
                    Ok(false) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected)
                ) {
                    registry.model_routes_status.lock().unwrap().watcher_error = true;
                    if matches!(event, Err(std::sync::mpsc::RecvTimeoutError::Disconnected)) {
                        watching = false;
                        registry.model_routes_status.lock().unwrap().watcher_active = false;
                    }
                }
                if event.is_ok() {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                // Retry a valid pending file when backend readiness changes.
                registry.refresh_model_routes();
            }
        });
    }

    pub fn workers_for_backends(&self, backends: &[String]) -> Vec<Arc<dyn Worker>> {
        backends
            .iter()
            .flat_map(|model| {
                self.get_workers_filtered(
                    Some(model),
                    Some(WorkerType::Regular),
                    Some(ConnectionMode::Http),
                    None,
                    false,
                )
                .to_vec()
            })
            .collect()
    }

    pub fn hash_ring_for_backends(&self, backends: &[String]) -> Arc<HashRing> {
        let generation = self.worker_generation.load(Ordering::Acquire);
        if let Some(entry) = self.alias_hash_rings.get(backends) {
            if entry.0 == generation {
                return entry.1.clone();
            }
        }
        let ring = Arc::new(HashRing::new(&self.workers_for_backends(backends)));
        if self.alias_hash_rings.len() >= 2048 {
            self.alias_hash_rings.clear();
        }
        self.alias_hash_rings
            .insert(backends.to_vec(), (generation, ring.clone()));
        ring
    }

    /// Rebuild the hash ring for a model based on current workers in the model index
    fn rebuild_hash_ring(&self, model_id: &str) {
        if let Some(workers) = self.model_index.get(model_id) {
            let ring = HashRing::new(&workers);
            self.hash_rings.insert(model_id.to_string(), Arc::new(ring));
        } else {
            // No workers for this model, remove the ring
            self.hash_rings.remove(model_id);
        }
    }

    /// Get the hash ring for a model (O(1) lookup)
    pub fn get_hash_ring(&self, model_id: &str) -> Option<Arc<HashRing>> {
        self.hash_rings.get(model_id).map(|r| Arc::clone(&r))
    }

    /// Set mesh sync manager (thread-safe, can be called after initialization)
    pub fn set_mesh_sync(&self, mesh_sync: OptionalMeshSyncManager) {
        *self.mesh_sync.write().unwrap() = mesh_sync;
    }

    /// Register a new worker
    pub fn register(&self, worker: Arc<dyn Worker>) -> WorkerId {
        let maintenance = self.maintenance.lock().unwrap();
        if let Some((id, _)) = maintenance.get(worker.url()) {
            return id.clone();
        }
        let worker_id = if let Some(existing_id) = self.url_to_id.get(worker.url()) {
            // Worker with this URL already exists, update it
            existing_id.clone()
        } else {
            WorkerId::new()
        };

        // Replace all indexes for an existing URL. Otherwise an old Arc can
        // remain selectable after a metadata update and fail admission forever.
        if self.workers.contains_key(&worker_id) {
            self.remove(&worker_id);
        }

        // Store worker
        self.workers.insert(worker_id.clone(), worker.clone());

        // Update URL mapping
        self.url_to_id
            .insert(worker.url().to_string(), worker_id.clone());

        // Update model index for O(1) lookups using copy-on-write
        // This creates a new immutable snapshot with the added worker
        let model_id = worker.model_id().to_string();
        self.model_index
            .entry(model_id.clone())
            .and_modify(|existing| {
                // Create new snapshot with the additional worker
                let mut new_workers: Vec<Arc<dyn Worker>> = existing.iter().cloned().collect();
                new_workers.push(worker.clone());
                *existing = Arc::from(new_workers.into_boxed_slice());
            })
            .or_insert_with(|| Arc::from(vec![worker.clone()].into_boxed_slice()));

        // Rebuild hash ring for this model
        self.rebuild_hash_ring(&model_id);
        self.worker_generation.fetch_add(1, Ordering::Release);

        // Update type index (clone needed for DashMap key ownership)
        self.type_workers
            .entry(worker.worker_type().clone())
            .or_default()
            .push(worker_id.clone());

        // Update connection mode index (clone needed for DashMap key ownership)
        self.connection_workers
            .entry(worker.connection_mode().clone())
            .or_default()
            .push(worker_id.clone());

        // Sync to mesh if enabled (no-op if mesh is not enabled)
        if let Some(ref mesh_sync) = *self.mesh_sync.read().unwrap() {
            mesh_sync.sync_worker_state(
                worker_id.as_str().to_string(),
                worker.model_id().to_string(),
                worker.url().to_string(),
                worker.is_healthy(),
                0.0, // TODO: Get actual load
            );
        }

        worker_id
    }

    pub fn is_draining_url(&self, url: &str) -> bool {
        self.maintenance.lock().unwrap().contains_key(url)
    }

    pub fn begin_drain(&self, id: &WorkerId) -> Option<(String, usize)> {
        let mut maintenance = self.maintenance.lock().unwrap();
        if let Some((url, (_, worker))) = maintenance.iter().find(|(_, (key, _))| key == id) {
            return Some((
                url.clone(),
                self.request_counts
                    .get(worker.url())
                    .map(|count| count.load(Ordering::Acquire))
                    .unwrap_or(0),
            ));
        }
        let worker = self.get(id)?;
        // The admission fence currently covers regular HTTP generation only.
        if worker.worker_type() != &WorkerType::Regular
            || worker.connection_mode() != &ConnectionMode::Http
        {
            return None;
        }
        let url = worker.url().to_string();
        maintenance.insert(url.clone(), (id.clone(), worker.clone()));
        self.remove(id);
        Some((
            url,
            self.request_counts
                .get(worker.url())
                .map(|count| count.load(Ordering::Acquire))
                .unwrap_or(0),
        ))
    }
    pub fn drain_status(&self, id: &WorkerId) -> Option<(String, usize)> {
        self.maintenance
            .lock()
            .unwrap()
            .iter()
            .find(|(_, (key, _))| key == id)
            .map(|(url, (_, worker))| {
                (
                    url.clone(),
                    self.request_counts
                        .get(worker.url())
                        .map(|count| count.load(Ordering::Acquire))
                        .unwrap_or(0),
                )
            })
    }
    pub fn finish_drain(&self, id: &WorkerId) -> bool {
        let mut maintenance = self.maintenance.lock().unwrap();
        let url = maintenance
            .iter()
            .find(|(_, (key, worker))| {
                key == id
                    && self
                        .request_counts
                        .get(worker.url())
                        .map(|count| count.load(Ordering::Acquire))
                        .unwrap_or(0)
                        == 0
            })
            .map(|(url, _)| url.clone());
        url.map(|url| maintenance.remove(&url).is_some())
            .unwrap_or(false)
    }
    pub fn admit_http_request(
        &self,
        worker: Arc<dyn Worker>,
        headers: Option<&http::HeaderMap>,
    ) -> Option<crate::core::WorkerLoadGuard> {
        let maintenance = self.maintenance.lock().unwrap();
        if maintenance.contains_key(worker.url())
            || !self
                .get_by_url(worker.url())
                .is_some_and(|current| Arc::ptr_eq(&current, &worker))
        {
            return None;
        }
        let count = self
            .request_counts
            .entry(worker.url().to_string())
            .or_insert_with(|| Arc::new(AtomicUsize::new(0)))
            .clone();
        Some(crate::core::WorkerLoadGuard::with_request_counter(
            worker, headers, count,
        ))
    }

    /// Reserve (or retrieve) a stable UUID for a worker URL.
    /// Uses atomic entry API to avoid race conditions between check and insert.
    pub fn reserve_id_for_url(&self, url: &str) -> WorkerId {
        self.url_to_id.entry(url.to_string()).or_default().clone()
    }

    /// Best-effort lookup of the URL for a given worker ID.
    pub fn get_url_by_id(&self, worker_id: &WorkerId) -> Option<String> {
        if let Some(worker) = self.get(worker_id) {
            return Some(worker.url().to_string());
        }
        self.url_to_id
            .iter()
            .find_map(|entry| (entry.value() == worker_id).then(|| entry.key().clone()))
    }

    /// Remove a worker by ID
    pub fn remove(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker)) = self.workers.remove(worker_id) {
            // Remove from URL mapping
            self.url_to_id.remove(worker.url());

            // Remove from model index using copy-on-write
            // Create new snapshot without the removed worker
            let worker_url = worker.url();
            let model_id = worker.model_id().to_string();
            if let Some(mut entry) = self.model_index.get_mut(&model_id) {
                let new_workers: Vec<Arc<dyn Worker>> = entry
                    .iter()
                    .filter(|w| w.url() != worker_url)
                    .cloned()
                    .collect();
                *entry = Arc::from(new_workers.into_boxed_slice());
            }

            // Rebuild hash ring for this model
            self.rebuild_hash_ring(&model_id);
            self.worker_generation.fetch_add(1, Ordering::Release);

            // Remove from type index
            if let Some(mut type_workers) = self.type_workers.get_mut(worker.worker_type()) {
                type_workers.retain(|id| id != worker_id);
            }

            // Remove from connection mode index
            if let Some(mut conn_workers) =
                self.connection_workers.get_mut(worker.connection_mode())
            {
                conn_workers.retain(|id| id != worker_id);
            }

            worker.set_healthy(false);
            Metrics::remove_worker_metrics(worker.url());

            // Sync removal to mesh if enabled (no-op if mesh is not enabled)
            if let Some(ref mesh_sync) = *self.mesh_sync.read().unwrap() {
                mesh_sync.remove_worker_state(worker_id.as_str());
            }

            Some(worker)
        } else {
            None
        }
    }

    /// Remove a worker by URL
    pub fn remove_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker_id)) = self.url_to_id.remove(url) {
            self.remove(&worker_id)
        } else {
            None
        }
    }

    /// Get a worker by ID
    pub fn get(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        self.workers.get(worker_id).map(|entry| entry.clone())
    }

    /// Get a worker by URL
    pub fn get_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        self.url_to_id.get(url).and_then(|id| self.get(&id))
    }

    /// Empty worker slice constant for returning when no workers found
    const EMPTY_WORKERS: &'static [Arc<dyn Worker>] = &[];

    /// Get all workers for a model (O(1) optimized, lock-free)
    /// Returns an Arc to the immutable worker slice - just an atomic refcount bump.
    /// This is the fastest possible read path with zero contention.
    pub fn get_by_model(&self, model_id: &str) -> Arc<[Arc<dyn Worker>]> {
        self.model_index
            .get(model_id)
            .map(|workers| Arc::clone(&workers))
            .unwrap_or_else(|| Arc::from(Self::EMPTY_WORKERS))
    }

    /// Get all workers by worker type
    pub fn get_by_type(&self, worker_type: &WorkerType) -> Vec<Arc<dyn Worker>> {
        self.type_workers
            .get(worker_type)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Update worker health status and sync to mesh
    pub fn update_worker_health(&self, worker_id: &WorkerId, is_healthy: bool) {
        if let Some(worker) = self.workers.get(worker_id) {
            // Update worker health (if Worker trait has a method for this)
            // For now, we'll just sync to mesh

            // Sync to mesh if enabled (no-op if mesh is not enabled)
            if let Some(ref mesh_sync) = *self.mesh_sync.read().unwrap() {
                mesh_sync.sync_worker_state(
                    worker_id.as_str().to_string(),
                    worker.model_id().to_string(),
                    worker.url().to_string(),
                    is_healthy,
                    0.0, // TODO: Get actual load
                );
            }
        }
    }

    /// Get all prefill workers (regardless of bootstrap_port)
    pub fn get_prefill_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .filter_map(|entry| {
                let worker = entry.value();
                match worker.worker_type() {
                    WorkerType::Prefill { .. } => Some(worker.clone()),
                    _ => None,
                }
            })
            .collect()
    }

    /// Get all decode workers
    pub fn get_decode_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.get_by_type(&WorkerType::Decode)
    }

    /// Get all workers by connection mode
    pub fn get_by_connection(&self, connection_mode: &ConnectionMode) -> Vec<Arc<dyn Worker>> {
        self.connection_workers
            .get(connection_mode)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get the number of workers in the registry
    pub fn len(&self) -> usize {
        self.workers.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// Get all workers
    pub fn get_all(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get all workers with their IDs
    pub fn get_all_with_ids(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.workers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Get all worker URLs
    pub fn get_all_urls(&self) -> Vec<String> {
        self.workers
            .iter()
            .map(|entry| entry.value().url().to_string())
            .collect()
    }

    pub fn get_all_urls_with_api_key(&self) -> Vec<(String, Option<String>)> {
        self.workers
            .iter()
            .map(|entry| {
                (
                    entry.value().url().to_string(),
                    entry.value().api_key().clone(),
                )
            })
            .collect()
    }

    /// Get all model IDs with workers (lock-free)
    pub fn get_models(&self) -> Vec<String> {
        self.model_index
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get workers filtered by multiple criteria
    ///
    /// This method allows flexible filtering of workers based on:
    /// - model_id: Filter by specific model
    /// - worker_type: Filter by worker type (Regular, Prefill, Decode)
    /// - connection_mode: Filter by connection mode (Http, Grpc)
    /// - runtime_type: Filter by runtime type (Sglang, Vllm, External)
    /// - healthy_only: Only return healthy workers
    pub fn get_workers_filtered(
        &self,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
        runtime_type: Option<RuntimeType>,
        healthy_only: bool,
    ) -> Vec<Arc<dyn Worker>> {
        // Start with the most efficient collection based on filters
        // Use model index when possible as it's O(1) lookup
        let workers: Vec<Arc<dyn Worker>> = if let Some(model) = model_id {
            self.get_by_model(model).to_vec()
        } else {
            self.get_all()
        };

        // Apply remaining filters
        workers
            .into_iter()
            .filter(|w| {
                // Check worker_type if specified
                if let Some(ref wtype) = worker_type {
                    if *w.worker_type() != *wtype {
                        return false;
                    }
                }

                // Check connection_mode if specified (using matches for flexible gRPC matching)
                if let Some(ref conn) = connection_mode {
                    if !w.connection_mode().matches(conn) {
                        return false;
                    }
                }

                // Check runtime_type if specified
                if let Some(ref rt) = runtime_type {
                    if w.metadata().runtime_type != *rt {
                        return false;
                    }
                }

                // Check health if required
                if healthy_only && !w.is_healthy() {
                    return false;
                }

                true
            })
            .collect()
    }

    /// Get worker statistics (lock-free)
    pub fn stats(&self) -> WorkerRegistryStats {
        let total_workers = self.workers.len();
        // Count models directly instead of allocating Vec via get_models() (lock-free)
        let total_models = self
            .model_index
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .count();

        let mut healthy_count = 0;
        let mut total_load = 0;
        let mut regular_count = 0;
        let mut prefill_count = 0;
        let mut decode_count = 0;
        let mut http_count = 0;
        let mut grpc_count = 0;
        let mut cb_open_count = 0;
        let mut cb_half_open_count = 0;

        // Iterate DashMap directly to avoid cloning all workers via get_all()
        for entry in self.workers.iter() {
            let worker = entry.value();
            if worker.is_healthy() {
                healthy_count += 1;
            }
            total_load += self
                .request_counts
                .get(worker.url())
                .map(|count| count.load(Ordering::Acquire))
                .unwrap_or(0);

            match worker.worker_type() {
                WorkerType::Regular => regular_count += 1,
                WorkerType::Prefill { .. } => prefill_count += 1,
                WorkerType::Decode => decode_count += 1,
            }

            match worker.connection_mode() {
                ConnectionMode::Http => http_count += 1,
                ConnectionMode::Grpc { .. } => grpc_count += 1,
            }

            match worker.circuit_breaker().state() {
                CircuitState::Open => cb_open_count += 1,
                CircuitState::HalfOpen => cb_half_open_count += 1,
                CircuitState::Closed => {}
            }
        }

        WorkerRegistryStats {
            total_workers,
            total_models,
            healthy_workers: healthy_count,
            unhealthy_workers: total_workers.saturating_sub(healthy_count),
            total_load,
            regular_workers: regular_count,
            prefill_workers: prefill_count,
            decode_workers: decode_count,
            http_workers: http_count,
            grpc_workers: grpc_count,
            circuit_breaker_open: cb_open_count,
            circuit_breaker_half_open: cb_half_open_count,
        }
    }

    /// Get counts of regular and PD workers efficiently (O(1))
    /// This avoids the overhead of get_all() which allocates memory and iterates all workers
    pub fn get_worker_distribution(&self) -> (usize, usize) {
        // Use the existing type_workers index for O(1) lookup
        let regular_count = self
            .type_workers
            .get(&WorkerType::Regular)
            .map(|v| v.len())
            .unwrap_or(0);

        // Get total workers count efficiently from DashMap
        let total_workers = self.workers.len();

        // PD workers are any workers that are not Regular
        let pd_count = total_workers.saturating_sub(regular_count);

        (regular_count, pd_count)
    }

    /// Start a health checker for all workers in the registry
    /// This should be called once after the registry is populated with workers
    pub(crate) fn start_health_checker(&self, check_interval_secs: u64) -> HealthChecker {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let workers_ref = self.workers.clone();

        let handle = tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(check_interval_secs));

            loop {
                interval.tick().await;

                // Check for shutdown signal
                if shutdown_clone.load(Ordering::Acquire) {
                    tracing::debug!("Registry health checker shutting down");
                    break;
                }

                // Get all workers from registry
                let workers: Vec<Arc<dyn Worker>> = workers_ref
                    .iter()
                    .map(|entry| entry.value().clone())
                    .collect();

                // Perform health checks in parallel for better performance
                // This is especially important when there are many workers
                let health_futures: Vec<_> = workers
                    .iter()
                    .filter(|worker| !worker.metadata().health_config.disable_health_check)
                    .map(|worker| {
                        let worker = worker.clone();
                        async move {
                            let _ = worker.check_health_async().await;
                        }
                    })
                    .collect();
                futures::future::join_all(health_futures).await;
            }
        });

        HealthChecker::new(handle, shutdown)
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the worker registry
#[derive(Debug, Clone)]
pub struct WorkerRegistryStats {
    /// Total number of registered workers
    pub total_workers: usize,
    /// Number of unique models served
    pub total_models: usize,
    /// Number of workers passing health checks
    pub healthy_workers: usize,
    /// Number of workers failing health checks
    pub unhealthy_workers: usize,
    /// Sum of current load across all workers
    pub total_load: usize,
    /// Number of regular (non-PD) workers
    pub regular_workers: usize,
    /// Number of prefill workers (PD mode)
    pub prefill_workers: usize,
    /// Number of decode workers (PD mode)
    pub decode_workers: usize,
    /// Number of HTTP-connected workers
    pub http_workers: usize,
    /// Number of gRPC-connected workers
    pub grpc_workers: usize,
    /// Number of workers with circuit breaker in Open state (not accepting requests)
    pub circuit_breaker_open: usize,
    /// Number of workers with circuit breaker in HalfOpen state (testing recovery)
    pub circuit_breaker_half_open: usize,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::{circuit_breaker::CircuitBreakerConfig, BasicWorkerBuilder};

    #[cfg(unix)]
    #[test]
    fn model_route_directory_watcher_accepts_a_projection_swap_without_an_api_call() {
        use std::os::unix::fs::symlink;
        let directory = tempfile::tempdir().unwrap();
        for (name, revision) in [("..first", 1), ("..second", 2)] {
            std::fs::create_dir(directory.path().join(name)).unwrap();
            std::fs::write(directory.path().join(name).join("routes.json"),
                serde_json::json!({"schemaVersion":1,"revision":revision,"routes":{"public":["backend"]}}).to_string()).unwrap();
        }
        symlink("..first", directory.path().join("..data")).unwrap();
        symlink("..data/routes.json", directory.path().join("routes.json")).unwrap();
        let mut registry = WorkerRegistry::new();
        registry.model_routes_file = Some(directory.path().join("routes.json"));
        registry.register(Arc::new(
            BasicWorkerBuilder::new("http://ready:30000")
                .labels(HashMap::from([("model_id".into(), "backend".into())]))
                .build(),
        ));
        registry.refresh_model_routes();
        let registry = Arc::new(registry);
        registry.watch_model_routes();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while !registry.model_routes_status().watcher_active {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        symlink("..second", directory.path().join("..next")).unwrap();
        std::fs::rename(
            directory.path().join("..next"),
            directory.path().join("..data"),
        )
        .unwrap();
        while registry.model_routes_status().accepted_revision != 2 {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(registry.model_routes_status().error.is_none());
    }

    #[test]
    fn model_route_refresh_retries_pending_data_and_preserves_unchanged_rings() {
        let path = std::env::temp_dir().join(format!("ci-route-refresh-{}", Uuid::new_v4()));
        let mut registry = WorkerRegistry::new();
        registry.model_routes_file = Some(path.clone());
        std::fs::write(
            &path,
            br#"{"schemaVersion":1,"revision":3,"routes":{"public":["backend"]}}"#,
        )
        .unwrap();
        registry.refresh_model_routes();
        assert_eq!(registry.model_routes_status().desired_revision, Some(3));
        assert_eq!(registry.model_routes_status().accepted_revision, 0);
        assert!(registry.model_routes_status().error.is_some());
        registry.register(Arc::new(
            BasicWorkerBuilder::new("http://ready:30000")
                .labels(HashMap::from([("model_id".into(), "backend".into())]))
                .build(),
        ));
        registry.refresh_model_routes();
        assert_eq!(registry.model_routes_status().accepted_revision, 3);
        assert!(registry.model_routes_status().error.is_none());
        let ring = registry.hash_ring_for_backends(&["backend".into()]);
        std::fs::write(
            &path,
            br#"{"schemaVersion":1,"revision":7,"routes":{"public":["backend"]}}"#,
        )
        .unwrap();
        registry.refresh_model_routes();
        assert_eq!(registry.model_routes_status().accepted_revision, 7);
        assert!(Arc::ptr_eq(
            &ring,
            &registry.hash_ring_for_backends(&["backend".into()])
        ));
        std::fs::write(&path, b"invalid").unwrap();
        registry.refresh_model_routes();
        assert_eq!(registry.model_routes_status().accepted_revision, 7);
        assert!(registry.model_routes_status().error.is_some());
        // Restoring accepted bytes clears the failure without a route change.
        std::fs::write(
            &path,
            br#"{"schemaVersion":1,"revision":7,"routes":{"public":["backend"]}}"#,
        )
        .unwrap();
        registry.refresh_model_routes();
        assert!(registry.model_routes_status().error.is_none());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_model_routes_commit_checks_hash_revision_and_ready_backends() {
        use sha2::{Digest, Sha256};
        let path = std::env::temp_dir().join(format!("ci-route-commit-{}", Uuid::new_v4()));
        let mut registry = WorkerRegistry::new();
        registry.model_routes_file = Some(path.clone());
        let mut ids = Vec::new();
        for (url, model) in [
            ("http://old:30000", "model-old"),
            ("http://new:30000", "model-new"),
        ] {
            ids.push(
                registry.register(Arc::new(
                    BasicWorkerBuilder::new(url)
                        .labels(HashMap::from([("model_id".into(), model.into())]))
                        .build(),
                )),
            );
        }
        let write = |revision: u64, backends: Vec<&str>| {
            let bytes = serde_json::to_vec(&serde_json::json!({
                "schemaVersion": 1, "revision": revision, "routes": {"public": backends}
            }))
            .unwrap();
            std::fs::write(&path, &bytes).unwrap();
            format!("{:x}", Sha256::digest(&bytes))
        };
        let hash = write(1, vec!["model-old", "model-new"]);
        assert!(registry.reload_model_routes(0, "wrong-hash").is_err());
        assert!(registry.reload_model_routes(1, &hash).is_err());
        assert_eq!(registry.model_routes().revision, 0);
        assert_eq!(registry.reload_model_routes(0, &hash).unwrap().revision, 1);
        // An interrupted commit can be replayed without another route change.
        assert_eq!(registry.reload_model_routes(0, &hash).unwrap().revision, 1);
        assert!(registry.reload_model_routes(7, &hash).is_err());
        let frozen = registry.resolve_model_route(Some("public")).unwrap();
        assert_eq!(registry.workers_for_backends(&frozen).len(), 2);
        let unknown = write(2, vec!["missing"]);
        assert!(registry.reload_model_routes(1, &unknown).is_err());
        assert_eq!(registry.model_routes().revision, 1);
        let next = write(2, vec!["model-new"]);
        assert_eq!(registry.reload_model_routes(1, &next).unwrap().revision, 2);
        assert_eq!(
            registry.resolve_model_route(Some("public")).unwrap(),
            vec!["model-new"]
        );
        // Accepted requests keep their route snapshot when new requests change.
        assert_eq!(frozen, vec!["model-old", "model-new"]);
        // A restart obtains the committed model route from the same durable file.
        assert_eq!(ModelRoutes::read(&path).unwrap().0, registry.model_routes());
        registry.begin_drain(&ids[1]).unwrap();
        let unavailable = write(3, vec!["model-new"]);
        assert!(registry.reload_model_routes(2, &unavailable).is_err());
        assert_eq!(registry.model_routes().revision, 2);
        assert_eq!(registry.workers_for_backends(&frozen).len(), 1);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_drain_keeps_other_model_workers_and_never_falls_back() {
        let registry = WorkerRegistry::new();
        let old: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://old:30000")
                .labels(HashMap::from([("model_id".into(), "model-old".into())]))
                .build(),
        );
        let new: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://new:30000")
                .labels(HashMap::from([("model_id".into(), "model-new".into())]))
                .build(),
        );
        let old_id = registry.register(old);
        registry.register(new);
        registry.begin_drain(&old_id).unwrap();
        assert!(registry
            .get_workers_filtered(Some("model-old"), None, None, None, false)
            .is_empty());
        let workers = registry.get_workers_filtered(Some("model-new"), None, None, None, false);
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].url(), "http://new:30000");
        assert!(registry
            .get_workers_filtered(Some("unknown-model"), None, None, None, false)
            .is_empty());
    }

    #[test]
    fn test_admission_and_drain_share_a_fence() {
        let registry = Arc::new(WorkerRegistry::new());
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://race:30000").build());
        let id = registry.register(worker.clone());
        let barrier = Arc::new(std::sync::Barrier::new(17));
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let registry = registry.clone();
                let worker = worker.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    registry.admit_http_request(worker, None)
                })
            })
            .collect();
        barrier.wait();
        registry.begin_drain(&id).unwrap();
        let guards: Vec<_> = threads
            .into_iter()
            .filter_map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(registry.drain_status(&id).unwrap().1, guards.len());
        assert!(registry.admit_http_request(worker.clone(), None).is_none());
        drop(guards);
        assert_eq!(registry.drain_status(&id).unwrap().1, 0);
        assert!(registry.finish_drain(&id));
        // An old selection cannot be admitted even after the URL fence is cleared.
        assert!(registry.admit_http_request(worker, None).is_none());
    }

    #[test]
    fn test_drain_fences_selection_and_queued_registration() {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://drain:30000").build());
        let id = registry.register(worker.clone());
        let guard = registry.admit_http_request(worker.clone(), None).unwrap();
        assert_eq!(
            registry.begin_drain(&id),
            Some(("http://drain:30000".into(), 1))
        );
        assert!(registry.get(&id).is_none());
        assert!(registry.admit_http_request(worker.clone(), None).is_none());
        assert!(!registry.finish_drain(&id));
        assert_eq!(
            registry.register(Arc::new(
                BasicWorkerBuilder::new("http://drain:30000").build()
            )),
            id
        );
        assert!(registry.get_by_url("http://drain:30000").is_none());
        assert_eq!(registry.begin_drain(&id).unwrap().1, 1);
        drop(guard);
        assert_eq!(registry.drain_status(&id).unwrap().1, 0);
        assert!(registry.finish_drain(&id));
        assert!(registry.drain_status(&id).is_none());
    }

    #[test]
    fn test_drain_counts_all_response_bodies_across_worker_instances() {
        let registry = WorkerRegistry::new();
        let old: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://drain:30000").build());
        let id = registry.register(old.clone());
        let guard = registry.admit_http_request(old, None).unwrap();
        let replacement: Arc<dyn Worker> =
            Arc::new(BasicWorkerBuilder::new("http://drain:30000").build());
        registry.register(replacement);
        assert_eq!(registry.get_all().len(), 1);
        assert_eq!(
            registry.get_by_model(crate::core::UNKNOWN_MODEL_ID).len(),
            1
        );
        let response = crate::core::AttachedBody::wrap_response(
            axum::response::Response::new(axum::body::Body::from("pending body")),
            guard,
        );
        assert_eq!(registry.begin_drain(&id).unwrap().1, 1);
        assert!(!registry.finish_drain(&id));
        drop(response);
        assert_eq!(registry.drain_status(&id).unwrap().1, 0);
        assert!(registry.finish_drain(&id));
    }

    #[test]
    fn test_worker_registry() {
        let registry = WorkerRegistry::new();

        // Create a worker with labels
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "llama-3-8b".to_string());
        labels.insert("priority".to_string(), "50".to_string());
        labels.insert("cost".to_string(), "0.8".to_string());

        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register worker
        let worker_id = registry.register(Arc::from(worker));

        assert!(registry.get(&worker_id).is_some());
        assert!(registry.get_by_url("http://worker1:8080").is_some());
        assert_eq!(registry.get_by_model("llama-3-8b").len(), 1);
        assert_eq!(registry.get_by_type(&WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(&ConnectionMode::Http).len(), 1);

        let stats = registry.stats();
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.total_models, 1);

        // Remove worker
        registry.remove(&worker_id);
        assert!(registry.get(&worker_id).is_none());
    }

    #[test]
    fn test_model_index_fast_lookup() {
        let registry = WorkerRegistry::new();

        // Create workers for different models
        let mut labels1 = HashMap::new();
        labels1.insert("model_id".to_string(), "llama-3".to_string());
        let worker1: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels1)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels2 = HashMap::new();
        labels2.insert("model_id".to_string(), "llama-3".to_string());
        let worker2: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker2:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels2)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels3 = HashMap::new();
        labels3.insert("model_id".to_string(), "gpt-4".to_string());
        let worker3: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker3:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels3)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register workers
        registry.register(Arc::from(worker1));
        registry.register(Arc::from(worker2));
        registry.register(Arc::from(worker3));

        let llama_workers = registry.get_by_model("llama-3");
        assert_eq!(llama_workers.len(), 2);
        let urls: Vec<String> = llama_workers.iter().map(|w| w.url().to_string()).collect();
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));

        let gpt_workers = registry.get_by_model("gpt-4");
        assert_eq!(gpt_workers.len(), 1);
        assert_eq!(gpt_workers[0].url(), "http://worker3:8080");

        let unknown_workers = registry.get_by_model("unknown-model");
        assert_eq!(unknown_workers.len(), 0);

        registry.remove_by_url("http://worker1:8080");
        let llama_workers_after = registry.get_by_model("llama-3");
        assert_eq!(llama_workers_after.len(), 1);
        assert_eq!(llama_workers_after[0].url(), "http://worker2:8080");
    }
}
