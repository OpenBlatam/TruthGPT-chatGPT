defmodule TruthGPT.MixProject do
  use Mix.Project

  @version "1.0.0"
  @source_url "https://github.com/truthgpt/optimization_core"

  def project do
    [
      app: :truthgpt,
      version: @version,
      elixir: "~> 1.14",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      aliases: aliases(),
      deps: deps(),
      name: "TruthGPT",
      description: "High-performance, fault-tolerant LLM inference with Elixir",
      package: package(),
      docs: docs(),
      source_url: @source_url
    ]
  end

  def application do
    [
      mod: {TruthGPT.Application, []},
      extra_applications: [:logger, :runtime_tools, :crypto]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      # Phoenix & Web
      {:phoenix, "~> 1.7.10"},
      {:phoenix_pubsub, "~> 2.1"},
      {:phoenix_live_view, "~> 0.20.1"},
      {:plug_cowboy, "~> 2.6"},
      {:jason, "~> 1.4"},
      {:cors_plug, "~> 3.0"},

      # Distributed Systems
      {:libcluster, "~> 3.3"},
      {:horde, "~> 0.8"},
      {:syn, "~> 3.3"},

      # Telemetry & Monitoring
      {:telemetry, "~> 1.2"},
      {:telemetry_metrics, "~> 0.6"},
      {:telemetry_poller, "~> 1.0"},
      {:prometheus_ex, "~> 3.0"},
      {:prometheus_plugs, "~> 1.1"},

      # Utilities
      {:uuid, "~> 1.1"},
      {:nimble_options, "~> 1.0"},
      {:poolboy, "~> 1.5"},

      # Compression
      {:ezstd, "~> 1.0"},

      # NIFs for Python/Rust interop
      {:rustler, "~> 0.30", runtime: false},
      {:erlport, "~> 0.10"},

      # Testing
      {:ex_machina, "~> 2.7", only: :test},
      {:stream_data, "~> 0.6", only: :test},

      # Development
      {:ex_doc, "~> 0.30", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false}
    ]
  end

  defp aliases do
    [
      setup: ["deps.get"],
      "assets.deploy": ["cmd --cd assets npm run deploy"],
      test: ["test"]
    ]
  end

  defp package do
    [
      maintainers: ["TruthGPT Team"],
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "Documentation" => "https://hexdocs.pm/truthgpt"
      }
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md"],
      source_ref: "v#{@version}"
    ]
  end
end












