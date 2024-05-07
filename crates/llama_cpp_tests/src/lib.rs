//! Test harness for [`llama_cpp`][lcpp] and [`llama_cpp_sys`].
//!
//! This crate expects one or more `llama.cpp`-compatible GGUF models to be made available in
//! a directory specified in the `LLAMA_CPP_TEST_MODELS` environment variable.

#[cfg(test)]
mod tests {
    use std::io;
    use std::io::{Cursor, Write};
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use futures::StreamExt;
    use image::io::Reader;
    use image::ImageFormat;
    use tokio::select;
    use tokio::time::Instant;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    use llama_cpp::multimodal::LlavaModel;
    use llama_cpp::standard_sampler::StandardSampler;
    use llama_cpp::{
        CompletionHandle, EmbeddingsParams, LlamaModel, LlamaParams, SessionParams, TokensToStrings,
    };

    fn init_tracing() {
        static SUBSCRIBER_SET: AtomicBool = AtomicBool::new(false);

        if !SUBSCRIBER_SET.swap(true, Ordering::SeqCst) {
            let format = tracing_subscriber::fmt::layer().compact();
            let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or(
                tracing_subscriber::EnvFilter::default()
                    .add_directive(tracing_subscriber::filter::LevelFilter::INFO.into()),
            );

            tracing_subscriber::registry()
                .with(format)
                .with(filter)
                .init();
        }
    }

    async fn list_models(dir: impl AsRef<Path>) -> Vec<String> {
        let dir = dir.as_ref();

        if !dir.is_dir() {
            panic!("\"{}\" is not a directory", dir.to_string_lossy());
        }

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

    // TODO theres a concurrency issue with vulkan, look into it
    #[ignore]
    #[tokio::test]
    async fn load_models() {
        let dir = std::env::var("LLAMA_CPP_TEST_MODELS").unwrap_or_else(|_| {
            panic!(
                "LLAMA_CPP_TEST_MODELS environment variable not set. \
                Please set this to the directory containing one or more GGUF models."
            );
        });

        let models = list_models(dir).await;

        for model in models {
            println!("Loading model: {}", model);
            let _model = LlamaModel::load_from_file_async(model, LlamaParams::default())
                .await
                .expect("Failed to load model");
        }
    }

    // #[ignore]
    #[tokio::test]
    async fn execute_completions() {
        init_tracing();

        let dir = std::env::var("LLAMA_CPP_TEST_MODELS").unwrap_or_else(|_| {
            panic!(
                "LLAMA_CPP_TEST_MODELS environment variable not set. \
                Please set this to the directory containing one or more GGUF models."
            );
        });

        let models = list_models(dir).await;

        for model in models {
            let mut params = LlamaParams::default();

            if cfg!(any(feature = "vulkan", feature = "cuda", feature = "metal")) {
                params.n_gpu_layers = i32::MAX as u32;
            }

            let model = LlamaModel::load_from_file_async(model, params)
                .await
                .expect("Failed to load model");

            let params = SessionParams {
                n_ctx: 2048,
                ..Default::default()
            };

            println!("{:?}", params);

            let estimate = model.estimate_session_size(&params);
            println!(
                "Predict chat session size: Host {}MB, Device {}MB",
                estimate.host_memory / 1024 / 1024,
                estimate.device_memory / 1024 / 1024,
            );

            let mut session = model
                .create_session(params)
                .expect("Failed to create session");

            println!(
                "Real chat session size: Host {}MB",
                session.memory_size() / 1024 / 1024
            );

            session
                .advance_context_async("<|SYSTEM|>You are a helpful assistant.")
                .await
                .unwrap();
            session
                .advance_context_async(r"<|SYSTEM|>
The standard Lorem Ipsum passage, used since the 1500s

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Section 1.10.32 of de Finibus Bonorum et Malorum, written by Cicero in 45 BC

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?
1914 translation by H. Rackham

But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness. No one rejects, dislikes, or avoids pleasure itself, because it is pleasure, but because those who do not know how to pursue pleasure rationally encounter consequences that are extremely painful. Nor again is there anyone who loves or pursues or desires to obtain pain of itself, because it is pain, but because occasionally circumstances occur in which toil and pain can procure him some great pleasure. To take a trivial example, which of us ever undertakes laborious physical exercise, except to obtain some advantage from it? But who has any right to find fault with a man who chooses to enjoy a pleasure that has no annoying consequences, or one who avoids a pain that produces no resultant pleasure?
Section 1.10.33 of de Finibus Bonorum et Malorum, written by Cicero in 45 BC

At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga. Et harum quidem rerum facilis est et expedita distinctio. Nam libero tempore, cum soluta nobis est eligendi optio cumque nihil impedit quo minus id quod maxime placeat facere possimus, omnis voluptas assumenda est, omnis dolor repellendus. Temporibus autem quibusdam et aut officiis debitis aut rerum necessitatibus saepe eveniet ut et voluptates repudiandae sint et molestiae non recusandae. Itaque earum rerum hic tenetur a sapiente delectus, ut aut reiciendis voluptatibus maiores alias consequatur aut perferendis doloribus asperiores repellat.
1914 translation by H. Rackham

On the other hand, we denounce with righteous indignation and dislike men who are so beguiled and demoralized by the charms of pleasure of the moment, so blinded by desire, that they cannot foresee the pain and trouble that are bound to ensue; and equal blame belongs to those who fail in their duty through weakness of will, which is the same as saying through shrinking from toil and pain. These cases are perfectly simple and easy to distinguish. In a free hour, when our power of choice is untrammelled and when nothing prevents our being able to do what we like best, every pleasure is to be welcomed and every pain avoided. But in certain circumstances and owing to the claims of duty or the obligations of business it will frequently occur that pleasures have to be repudiated and annoyances accepted. The wise man therefore always holds in these matters to this principle of selection: he rejects pleasures to secure other greater pleasures, or else he endures pains to avoid worse pains.
")
                .await
                .unwrap();
            session
                .advance_context_async("<|USER|>Hello!")
                .await
                .unwrap();
            session
                .advance_context_async("<|ASSISTANT|>")
                .await
                .unwrap();

            let mut completions = session
                .start_completing_with(StandardSampler::default(), 1024)
                .expect("Failed to start completing")
                .into_strings();
            let timeout_by = Instant::now() + Duration::from_secs(500);

            println!();

            loop {
                select! {
                    _ = tokio::time::sleep_until(timeout_by) => {
                        break;
                    }
                    completion = <TokensToStrings<CompletionHandle> as StreamExt>::next(&mut completions) => {
                        if let Some(completion) = completion {
                            print!("{completion}");
                            let _ = io::stdout().flush();
                        } else {
                            break;
                        }
                        continue;
                    }
                }
            }
            println!();
            println!();
        }
    }

    #[ignore]
    #[tokio::test]
    async fn embed() {
        init_tracing();

        let dir = std::env::var("LLAMA_EMBED_MODELS_DIR").unwrap_or_else(|_| {
            panic!(
                "LLAMA_EMBED_MODELS_DIR environment variable not set. \
                Please set this to the directory containing one or more embedding GGUF models."
            );
        });

        let models = list_models(dir).await;

        for model in models {
            let params = LlamaParams::default();
            let model = LlamaModel::load_from_file_async(model, params)
                .await
                .expect("Failed to load model");

            let mut input = vec![];

            for _phrase_idx in 0..10 {
                let mut phrase = String::new();
                for _word_idx in 0..200 {
                    phrase.push_str("word ");
                }
                phrase.truncate(phrase.len() - 1);
                input.push(phrase);
            }

            let params = EmbeddingsParams::default();

            let tokenized_input = model
                .tokenize_slice(&input, true, false)
                .expect("Failed to tokenize input");
            let estimate = model.estimate_embeddings_session_size(&tokenized_input, &params);
            println!(
                "Predict embeddings session size: Host {}MB, Device {}MB",
                estimate.host_memory / 1024 / 1024,
                estimate.device_memory / 1024 / 1024,
            );

            let res = model
                .embeddings_async(&input, params)
                .await
                .expect("Failed to infer embeddings");

            println!("{:?}", res[0]);

            for embedding in &res {
                let mut sum = 0f32;
                for value in embedding {
                    assert!(value.is_normal(), "Embedding value isn't normal");
                    assert!(*value >= -1f32, "Embedding value isn't normalised");
                    assert!(*value <= 1f32, "Embedding value isn't normalised");
                    sum += value * value;
                }

                const ERROR: f32 = 0.0001;
                let mag = sum.sqrt();
                assert!(mag < 1. + ERROR, "Vector magnitude is not close to 1");
                assert!(mag > 1. - ERROR, "Vector magnitude is not close to 1");
            }
        }
    }

    #[ignore]
    #[tokio::test]
    async fn llava() {
        init_tracing();

        let dir = std::env::var("LLAMA_CPP_TEST_MODELS").unwrap_or_else(|_| {
            panic!(
                "LLAMA_CPP_TEST_MODELS environment variable not set. \
                Please set this to the directory containing one or more GGUF models."
            );
        });

        let mmproj_path = std::env::var("LLAMA_CPP_TEST_MMPROJ").unwrap_or_else(|_| {
            panic!(
                "LLAMA_CPP_TEST_MMPROJ environment variable not set. \
                Please set this to the path of a multimodal projector GGUF file."
            );
        });

        let image_path = std::env::var("LLAMA_CPP_TEST_IMAGE").unwrap_or_else(|_| {
            panic!(
                "LLAMA_CPP_TEST_IMAGE environment variable not set. \
                Please set this to the path of an image file."
            );
        });

        let img = {
            let reader = Reader::open(image_path)
                .expect("Failed to open image")
                .decode()
                .expect("Failed to decode image");

            // Llama.cpp only accepts encoded images
            let bytes = vec![];
            let mut cursor = Cursor::new(bytes);
            reader
                .write_to(&mut cursor, ImageFormat::Png)
                .expect("Failed to re-encode image");
            cursor.into_inner()
        };

        let models = list_models(dir).await;

        for model in models {
            let params = LlamaParams::default();

            let model = LlavaModel::load_from_file_async(model, params, &mmproj_path)
                .await
                .expect("Failed to load model");

            let params = SessionParams {
                n_ctx: 2048,
                ..Default::default()
            };

            let mut session = model
                .create_session(params)
                .expect("Failed to create session");

            session
                .advance_context_async("<|SYSTEM|>You are a helpful assistant.")
                .await
                .unwrap();
            session
                .advance_with_image(&img)
                .expect("Failed to advance context with image");
            session
                .advance_context_async("<|USER|>Provide a full description.")
                .await
                .unwrap();
            session
                .advance_context_async("<|ASSISTANT|>")
                .await
                .unwrap();

            let mut completions = session
                .start_completing_with(StandardSampler::default(), 1024)
                .expect("Failed to start completing")
                .into_strings();
            let timeout_by = Instant::now() + Duration::from_secs(500);

            println!();

            loop {
                select! {
                    _ = tokio::time::sleep_until(timeout_by) => {
                        break;
                    }
                    completion = <TokensToStrings<CompletionHandle> as StreamExt>::next(&mut completions) => {
                        if let Some(completion) = completion {
                            print!("{completion}");
                            let _ = io::stdout().flush();
                        } else {
                            break;
                        }
                        continue;
                    }
                }
            }
            println!();
            println!();
        }
    }
}
