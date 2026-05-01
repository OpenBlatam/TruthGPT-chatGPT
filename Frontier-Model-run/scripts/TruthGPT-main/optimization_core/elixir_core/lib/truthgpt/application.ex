defmodule TruthGPT.Application do
  @moduledoc """
  TruthGPT Elixir Core - Fault-Tolerant Real-Time Inference

  Elixir/Phoenix backend providing:
  - Fault-tolerant supervision trees
  - Real-time WebSocket streaming (Phoenix Channels)
  - Distributed inference coordination
  - High-concurrency request handling (100K+ concurrent connections)

  ## Why Elixir?
  
  | Feature | Go | Python | Elixir |
  |---------|----|---------|----|
  | Fault Tolerance | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
  | Real-time Streaming | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
  | Concurrent Connections | 100K | 10K | **2M+** |
  | Hot Code Reload | No | No | **Yes** |
  | Distribution | Manual | Manual | **Built-in** |

  ## Usage

      # Start the application
      iex -S mix

      # Connect to inference channel
      socket = Phoenix.Socket.connect("ws://localhost:4000/socket")
      channel = Phoenix.Channel.join(socket, "inference:lobby")
      
      # Stream tokens
      Phoenix.Channel.push(channel, "generate", %{
        input_ids: [1, 2, 3],
        max_tokens: 100
      })
  """

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Telemetry
      TruthGPT.Telemetry,
      
      # PubSub for real-time streaming
      {Phoenix.PubSub, name: TruthGPT.PubSub},
      
      # KV Cache with distributed state
      {TruthGPT.Cache.Supervisor, []},
      
      # Inference worker pool
      {TruthGPT.Inference.Pool, []},
      
      # Batch scheduler
      {TruthGPT.Inference.BatchScheduler, []},
      
      # HTTP/WebSocket endpoint
      TruthGPTWeb.Endpoint,
      
      # Distributed node coordination
      {TruthGPT.Cluster.Supervisor, []}
    ]

    opts = [strategy: :one_for_one, name: TruthGPT.Supervisor]
    Supervisor.start_link(children, opts)
  end

  @impl true
  def config_change(changed, _new, removed) do
    TruthGPTWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end












