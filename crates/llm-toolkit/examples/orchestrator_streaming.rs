use llm_toolkit::{
    agent::impls::ClaudeCodeAgent,
    orchestrator::{BlueprintWorkflow, Orchestrator},
};
use serde_json::Value;
use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{Event, Subscriber};
use tracing_subscriber::prelude::*;
use tracing_subscriber::{Layer, layer::Context};

// A custom layer that sends tracing events to a channel as JSON strings.
pub struct JsonChannelLayer {
    sender: mpsc::Sender<String>,
}

impl<S> Layer<S> for JsonChannelLayer
where
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let mut fields = HashMap::new();
        let mut visitor = JsonVisitor(&mut fields);
        event.record(&mut visitor);

        let mut span_fields = HashMap::new();
        if let Some(span) = ctx.current_span().id().and_then(|id| ctx.span(id)) {
            let extensions = span.extensions();
            if let Some(visitor) = extensions.get::<JsonVisitor>() {
                span_fields = visitor.0.clone();
            }
        }

        let output = serde_json::json!({
            "target": event.metadata().target(),
            "level": event.metadata().level().to_string(),
            "message": fields.get("message").cloned().unwrap_or_default(),
            "fields": fields,
            "span": span_fields,
        });

        // Non-blocking send
        let _ = self.sender.try_send(output.to_string());
    }
}

// A visitor that extracts fields from a tracing event into a HashMap.
struct JsonVisitor<'a>(&'a mut HashMap<String, Value>);

impl<'a> tracing::field::Visit for JsonVisitor<'a> {
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.0
            .insert(field.name().to_string(), serde_json::json!(value));
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.0
            .insert(field.name().to_string(), serde_json::json!(value));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.0
            .insert(field.name().to_string(), serde_json::json!(value));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.0
            .insert(field.name().to_string(), serde_json::json!(value));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0
            .insert(field.name().to_string(), serde_json::json!(value));
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.insert(
            field.name().to_string(),
            serde_json::json!(format!("{:?}", value)),
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (tx, mut rx) = mpsc::channel::<String>(100);

    // Initialize the tracing subscriber with our custom layer
    let subscriber = tracing_subscriber::registry()
        .with(JsonChannelLayer { sender: tx })
        .with(tracing_subscriber::filter::EnvFilter::new("info")); // Set default log level
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    // Spawn a task to listen for and print events from the channel
    let listener_handle = tokio::spawn(async move {
        println!("--- Orchestrator Events ---");
        while let Some(event_str) = rx.recv().await {
            // Pretty-print the JSON
            match serde_json::from_str::<Value>(&event_str) {
                Ok(json) => {
                    if let Ok(pretty_json) = serde_json::to_string_pretty(&json) {
                        println!("{}", pretty_json);
                        println!("--------------------");
                    }
                }
                Err(_) => {
                    println!("{}", event_str); // Fallback for non-JSON
                    println!("--------------------");
                }
            }
        }
    });

    // --- Orchestrator Setup ---
    let blueprint = BlueprintWorkflow::new(
        "Goal: Write a short, engaging blog post about a technical topic.

        Workflow:
        1.  **Brainstorming**: Generate a few potential topics.
        2.  **Outline Creation**: Choose the best topic and create an outline.
        3.  **Drafting**: Write the blog post based on the outline.
        4.  **Review and Refine**: Review the draft for clarity, tone, and technical accuracy.
    "
        .to_string(),
    );

    let mut orchestrator = Orchestrator::new(blueprint);
    orchestrator.add_agent(ClaudeCodeAgent::new());

    // --- Execute Orchestrator ---
    let task = "the Rust programming language's ownership system";
    let result = orchestrator.execute(task).await;

    // --- Print Final Result ---
    println!("\n--- Final Orchestrator Result ---");
    println!("{}", serde_json::to_string_pretty(&result)?);

    // Ensure the listener has time to process all events
    // In a real app, you'd manage the channel closing more gracefully.
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    drop(orchestrator); // Drop orchestrator to ensure all spans are closed.

    // The sender is dropped, so the receiver will stop.
    listener_handle.await?;

    Ok(())
}
