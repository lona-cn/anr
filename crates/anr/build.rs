fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-env-changed=ANR_MODEL_PATH");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    #[cfg(any(feature = "download-models", feature = "copy-models"))]
    if std::env::var_os("DOCS_RS").is_none() {
        models::prepare()?;
    }
    Ok(())
}

#[cfg(any(feature = "download-models", feature = "copy-models"))]
mod models {
    use std::{env, fs, path::PathBuf};

    type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

    pub fn prepare() -> Result<()> {
        let model = match env::var_os("ANR_MODEL_PATH") {
            Some(path) => PathBuf::from(path),
            None => download()?,
        };
        let model = model
            .canonicalize()
            .map_err(|error| format!("cannot read DFN3 model {}: {error}", model.display()))?;
        if !model.is_file() {
            return Err(format!("DFN3 model is not a file: {}", model.display()).into());
        }
        println!("cargo:rerun-if-changed={}", model.display());
        println!(
            "cargo:rustc-env=ANR_DEFAULT_MODEL_PATH={}",
            model
                .to_str()
                .ok_or("DFN3 model path must be valid UTF-8")?
        );
        #[cfg(feature = "copy-models")]
        copy(&model)?;
        Ok(())
    }

    #[cfg(not(feature = "download-models"))]
    fn download() -> Result<PathBuf> {
        Err("copy-models requires ANR_MODEL_PATH or the download-models feature".into())
    }

    #[cfg(feature = "copy-models")]
    fn copy(model: &std::path::Path) -> Result<()> {
        let out_dir = PathBuf::from(env::var_os("OUT_DIR").ok_or("OUT_DIR is not set")?);
        // Derive the profile directory from OUT_DIR, including custom targets/profiles.
        let target_dir = out_dir
            .ancestors()
            .find(|path| path.ends_with("build"))
            .and_then(std::path::Path::parent)
            .ok_or("cannot determine Cargo output directory from OUT_DIR")?;
        for directory in [
            target_dir.to_path_buf(),
            target_dir.join("examples"),
            target_dir.join("deps"),
        ] {
            let destination = directory.join("models/dfn3.onnx");
            fs::create_dir_all(destination.parent().unwrap())?;
            if destination.canonicalize().ok().as_deref() != Some(model) {
                fs::copy(model, &destination)?;
            }
            println!("cargo:rerun-if-changed={}", destination.display());
        }
        Ok(())
    }

    #[cfg(feature = "download-models")]
    const MODEL_URL: &str =
        "https://github.com/lona-cn/anr/releases/download/model-dfn3-fe5eb64fa2e4/dfn3.onnx";
    #[cfg(feature = "download-models")]
    const MODEL_SHA256: [u8; 32] = [
        0xfe, 0x5e, 0xb6, 0x4f, 0xa2, 0xe4, 0x15, 0x4c, 0x83, 0xf8, 0xe4, 0x93, 0x5e, 0x82, 0x87,
        0x1c, 0x85, 0x0c, 0x15, 0x43, 0x87, 0xee, 0x89, 0x2e, 0x0a, 0xb6, 0x5f, 0xe1, 0x79, 0xe7,
        0xd8, 0xc9,
    ];

    #[cfg(feature = "download-models")]
    fn download() -> Result<PathBuf> {
        use std::io;
        use ureq::tls::{RootCerts, TlsConfig, TlsProvider};

        for variable in ["CARGO_HOME", "HOME", "USERPROFILE", "CARGO_NET_OFFLINE"] {
            println!("cargo:rerun-if-env-changed={variable}");
        }
        let cargo_home = env::var_os("CARGO_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
                    .map(|home| PathBuf::from(home).join(".cargo"))
            })
            .ok_or("cannot determine Cargo cache directory; set CARGO_HOME or ANR_MODEL_PATH")?;
        let cache = cargo_home.join("anr/models/dfn3-fe5eb64fa2e4");
        let model = cache.join("dfn3.onnx");
        if model.is_file() {
            copy_verified(fs::File::open(&model)?, io::sink())?;
            return Ok(model);
        }
        if env::var("CARGO_NET_OFFLINE")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        {
            return Err("DFN3 is not cached and CARGO_NET_OFFLINE is set; provide ANR_MODEL_PATH or download the model once with network access".into());
        }
        fs::create_dir_all(&cache)?;
        let mut temporary = tempfile::NamedTempFile::new_in(&cache)?;
        let agent = ureq::Agent::new_with_config(
            ureq::Agent::config_builder()
                .https_only(true)
                .max_redirects(3)
                .tls_config(
                    TlsConfig::builder()
                        .provider(TlsProvider::NativeTls)
                        .root_certs(RootCerts::PlatformVerifier)
                        .build(),
                )
                .timeout_global(Some(std::time::Duration::from_secs(300)))
                .build(),
        );
        println!("cargo:warning=Downloading DFN3 model from {MODEL_URL}");
        let response = agent.get(MODEL_URL).call().map_err(|error| {
            format!("failed to download DFN3 from {MODEL_URL}: {error}; set ANR_MODEL_PATH to use a local model")
        })?;
        let reader = response
            .into_body()
            .into_with_config()
            .limit(32 * 1024 * 1024)
            .reader();
        copy_verified(reader, &mut temporary)?;
        match temporary.persist_noclobber(&model) {
            Ok(_) => {}
            Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => {
                // Another build may have populated the shared cache concurrently.
                copy_verified(fs::File::open(&model)?, io::sink())?;
            }
            Err(error) => return Err(error.into()),
        }
        Ok(model)
    }

    #[cfg(feature = "download-models")]
    fn copy_verified(
        mut reader: impl std::io::Read,
        mut writer: impl std::io::Write,
    ) -> Result<()> {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        let mut buffer = [0; 64 * 1024];
        loop {
            let count = reader.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
            writer.write_all(&buffer[..count])?;
        }
        let digest = hasher.finalize();
        if digest[..] != MODEL_SHA256 {
            return Err(format!(
                "DFN3 model SHA-256 mismatch ({digest:x}); expected the pinned model from {MODEL_URL}"
            )
            .into());
        }
        Ok(())
    }
}
