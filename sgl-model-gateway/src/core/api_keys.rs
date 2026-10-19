//! Runtime-mintable API keys with coarse scopes.
//!
//! The router historically carried exactly two credentials: the inference
//! `api_key` and `--admin-api-key`. Letting a partner retune the admission
//! ceiling therefore meant handing over the admin key, which also reaches
//! `/admin/metrics` and can silently stop metric recording. This store lets the
//! admin mint narrower credentials instead of sharing its own.
//!
//! Keys are held as SHA-256 digests. The plaintext is returned exactly once, at
//! mint time, and is not recoverable afterwards: a lost key is revoked and
//! re-minted, not looked up.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use parking_lot::RwLock;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tracing::{info, warn};

/// What a credential may do. `Admin` satisfies every other scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// Read and set the admission ceiling. Nothing else.
    Admission,
    /// Everything, including metric recording and minting further keys.
    Admin,
}

impl Scope {
    /// Whether a credential carrying `self` may pass a route requiring `required`.
    pub fn satisfies(self, required: Scope) -> bool {
        self == Scope::Admin || self == required
    }
}

/// A minted key as stored. The plaintext is deliberately absent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyRecord {
    pub id: String,
    pub name: String,
    pub scope: Scope,
    /// Hex SHA-256 of the plaintext. Never the plaintext itself.
    pub hash: String,
    pub created_at: String,
}

fn hash_token(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn random_suffix() -> String {
    let mut bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Hashed, optionally file-backed store of minted credentials.
#[derive(Debug)]
pub struct KeyStore {
    records: RwLock<HashMap<String, KeyRecord>>,
    /// Rewritten after every mutation; without it a restart silently
    /// invalidates every key a partner holds.
    path: Option<PathBuf>,
}

impl KeyStore {
    /// Open a store, loading any existing records from `path`. A corrupt file
    /// is logged and treated as empty; refusing to boot over a credential file
    /// would take the router down with it.
    pub fn open(path: Option<PathBuf>) -> Self {
        let records = match path.as_deref() {
            Some(p) if p.exists() => match Self::load(p) {
                Ok(loaded) => {
                    info!("Loaded {} API key record(s) from {}", loaded.len(), p.display());
                    loaded
                }
                Err(err) => {
                    warn!("Could not read API key store {}: {err}; starting empty", p.display());
                    HashMap::new()
                }
            },
            _ => HashMap::new(),
        };
        Self {
            records: RwLock::new(records),
            path,
        }
    }

    fn load(path: &Path) -> Result<HashMap<String, KeyRecord>, String> {
        let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let list: Vec<KeyRecord> = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        Ok(list.into_iter().map(|r| (r.id.clone(), r)).collect())
    }

    /// Temp file plus rename, so an interrupted write leaves the store intact.
    fn persist(&self, records: &HashMap<String, KeyRecord>) {
        let Some(path) = self.path.as_deref() else {
            return;
        };
        let list: Vec<&KeyRecord> = records.values().collect();
        let Ok(encoded) = serde_json::to_string_pretty(&list) else {
            warn!("Could not encode API key store");
            return;
        };
        let tmp = path.with_extension("tmp");
        if let Err(err) = std::fs::write(&tmp, encoded) {
            warn!("Could not write API key store {}: {err}", tmp.display());
            return;
        }
        if let Err(err) = std::fs::rename(&tmp, path) {
            warn!("Could not replace API key store {}: {err}", path.display());
        }
    }

    /// Mint a credential; the returned plaintext is not recoverable afterwards.
    /// `explicit` pins a readable value (`sk-m3-minimax`) over a random one,
    /// trading entropy for legibility on an already-admin-gated call.
    pub fn mint(&self, name: &str, scope: Scope, explicit: Option<String>) -> (KeyRecord, String) {
        let plaintext = explicit.unwrap_or_else(|| format!("sk-smg-{}", random_suffix()));
        let record = KeyRecord {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            scope,
            hash: hash_token(&plaintext),
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        {
            let mut records = self.records.write();
            records.insert(record.id.clone(), record.clone());
            self.persist(&records);
        }
        info!("Minted API key '{}' ({:?}) id={}", name, scope, record.id);
        (record, plaintext)
    }

    /// Whether `token` is a live key whose scope satisfies `required`.
    pub fn verify(&self, token: &str, required: Scope) -> bool {
        self.identify(token, required).is_some()
    }

    /// The record `token` authenticates as, so a control-plane change can name
    /// the credential that made it. Keys are shared with partners, so "who"
    /// is not recoverable after the fact from anything else.
    pub fn identify(&self, token: &str, required: Scope) -> Option<KeyRecord> {
        let digest = hash_token(token);
        let records = self.records.read();
        records
            .values()
            .find(|record| {
                record.scope.satisfies(required)
                    && record.hash.as_bytes().ct_eq(digest.as_bytes()).unwrap_u8() == 1
            })
            .cloned()
    }

    pub fn list(&self) -> Vec<KeyRecord> {
        let mut out: Vec<KeyRecord> = self.records.read().values().cloned().collect();
        out.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        out
    }

    /// Revoke by id. Returns whether a record was actually removed.
    pub fn revoke(&self, id: &str) -> bool {
        let mut records = self.records.write();
        let removed = records.remove(id).is_some();
        if removed {
            self.persist(&records);
            info!("Revoked API key id={id}");
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admin_satisfies_every_scope() {
        assert!(Scope::Admin.satisfies(Scope::Admission));
        assert!(Scope::Admin.satisfies(Scope::Admin));
    }

    #[test]
    fn admission_does_not_reach_admin() {
        assert!(Scope::Admission.satisfies(Scope::Admission));
        assert!(!Scope::Admission.satisfies(Scope::Admin));
    }

    #[test]
    fn minted_key_verifies_only_within_its_scope() {
        let store = KeyStore::open(None);
        let (_, key) = store.mint("partner", Scope::Admission, None);
        assert!(store.verify(&key, Scope::Admission));
        assert!(!store.verify(&key, Scope::Admin));
        assert!(!store.verify("sk-not-a-key", Scope::Admission));
    }

    #[test]
    fn explicit_value_is_honoured_and_revocable() {
        let store = KeyStore::open(None);
        let (record, key) = store.mint("minimax", Scope::Admission, Some("sk-m3-minimax".into()));
        assert_eq!(key, "sk-m3-minimax");
        assert!(store.verify(&key, Scope::Admission));
        assert!(store.revoke(&record.id));
        assert!(!store.verify(&key, Scope::Admission));
        assert!(!store.revoke(&record.id));
    }

    #[test]
    fn plaintext_is_not_retained() {
        let store = KeyStore::open(None);
        let (record, key) = store.mint("partner", Scope::Admission, None);
        assert_ne!(record.hash, key);
        assert!(store.list().iter().all(|r| r.hash != key));
    }

    #[test]
    fn records_survive_a_reopen() {
        let dir = std::env::temp_dir().join(format!("smg-keys-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("keys.json");

        let key = {
            let store = KeyStore::open(Some(path.clone()));
            let (_, key) = store.mint("partner", Scope::Admission, None);
            key
        };

        let reopened = KeyStore::open(Some(path));
        assert!(reopened.verify(&key, Scope::Admission));
        std::fs::remove_dir_all(&dir).ok();
    }
}
