//! Test harness for [`llama_cpp`][lcpp] and [`llama_cpp_sys`].
//!
//! This crate expects one or more `llama.cpp`-compatible GGUF models to be made available in
//! a directory specified in the `LLAMA_CPP_TEST_MODELS` environment variable.

#[cfg(test)]
mod tests {
    use llama_cpp::LlamaModel;
    use std::time::Duration;
    use tokio::select;
    use tokio::time::Instant;

    async fn list_models() -> Vec<String> {
        let mut dir = std::env::var("LLAMA_CPP_TEST_MODELS").unwrap_or_else(|_| {
            eprintln!(
                "LLAMA_CPP_TEST_MODELS environment variable not set. \
                Please set this to the directory containing one or more GGUF models."
            );

            std::process::exit(1)
        });

        if !dir.ends_with('/') {
            dir.push('/');
        }

        let dir = std::path::Path::new(&dir);
        let mut models = tokio::fs::read_dir(dir).await.unwrap();
        let mut rv = vec![];

        while let Some(model) = models.next_entry().await.unwrap() {
            let path = model.path();

            if path.is_file() {
                let path = path.to_str().unwrap();
                if path.ends_with(".gguf") {
                    rv.push(path.to_string());
                }
            }
        }

        rv
    }

    #[tokio::test]
    async fn load_models() {
        let models = list_models().await;

        for model in models {
            println!("Loading model: {}", model);
            let _model = LlamaModel::load_from_file_async(model).await.unwrap();
        }
    }

    #[tokio::test]
    async fn execute_completions() {
        let models = list_models().await;

        for model in models {
            let model = LlamaModel::load_from_file_async(model).await.unwrap();
            let mut session = model.create_session();

            let mut completions = session.start_completing();
            let timeout_by = Instant::now() + Duration::from_secs(5);

            loop {
                select! {
                    _ = tokio::time::sleep_until(timeout_by) => {
                        break;
                    }
                    _completion = completions.next_token_async() => {
                        continue;
                    }
                }
            }
        }
    }
}
