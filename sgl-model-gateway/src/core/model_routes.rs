//! Versioned model aliases. The mounted file is the durable source.
use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelRoutes {
    pub schema_version: u32,
    pub revision: u64,
    pub routes: BTreeMap<String, Vec<String>>,
}

impl Default for ModelRoutes {
    fn default() -> Self {
        Self {
            schema_version: 1,
            revision: 0,
            routes: BTreeMap::new(),
        }
    }
}

impl ModelRoutes {
    pub fn read(path: &Path) -> Result<(Self, String), String> {
        let root = std::fs::canonicalize(path.parent().unwrap_or_else(|| Path::new(".")))
            .map_err(|_| "model route directory is unavailable")?;
        let target = std::fs::canonicalize(path).map_err(|_| "model route file is unavailable")?;
        if target == root || !target.starts_with(&root) {
            return Err("model route file leaves its directory".into());
        }
        let safe = |metadata: &std::fs::Metadata| {
            metadata.is_file() && metadata.len() > 0 && metadata.len() <= 65_536
        };
        if !safe(
            &std::fs::symlink_metadata(&target).map_err(|_| "model route file is unavailable")?,
        ) {
            return Err("model route file has an unsafe shape".into());
        }
        let file = File::open(target).map_err(|_| "model route file is unavailable")?;
        if !safe(
            &file
                .metadata()
                .map_err(|_| "model route file is unavailable")?,
        ) {
            return Err("model route file has an unsafe shape".into());
        }
        let mut bytes = Vec::new();
        file.take(65_537)
            .read_to_end(&mut bytes)
            .map_err(|_| "model route file read failed")?;
        if bytes.len() > 65_536 {
            return Err("model route file exceeds its limit".into());
        }
        let document: Self =
            serde_json::from_slice(&bytes).map_err(|_| "invalid model route document")?;
        if document.schema_version != 1
            || document.routes.len() > 1024
            || document.routes.iter().any(|(a, b)| {
                b.is_empty()
                    || b.len() > 1024
                    || b.iter().collect::<std::collections::HashSet<_>>().len() != b.len()
                    || std::iter::once(a)
                        .chain(b.iter())
                        .any(|s| s.is_empty() || s.len() > 256 || s.chars().any(char::is_control))
            })
        {
            return Err("invalid model route fields".into());
        }
        Ok((document, format!("{:x}", Sha256::digest(&bytes))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn read_is_bounded_and_preserves_the_versioned_snapshot() {
        let path = std::env::temp_dir().join(format!("ci-model-routes-{}", uuid::Uuid::new_v4()));
        let raw = br#"{"schemaVersion":1,"revision":3,"routes":{"public":["backend-old","backend-new"]}}"#;
        std::fs::write(&path, raw).unwrap();
        let (document, hash) = ModelRoutes::read(&path).unwrap();
        assert_eq!(document.revision, 3);
        assert_eq!(
            document.routes["public"],
            vec!["backend-old", "backend-new"]
        );
        assert_eq!(hash, format!("{:x}", Sha256::digest(raw)));
        // A restart reads the same durable snapshot, without an API replay.
        assert_eq!(ModelRoutes::read(&path).unwrap().0, document);
        std::fs::write(&path, vec![b' '; 65_537]).unwrap();
        assert!(ModelRoutes::read(&path).is_err());
        std::fs::write(&path, br#"{"schemaVersion":2,"revision":3,"routes":{}}"#).unwrap();
        assert!(ModelRoutes::read(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
