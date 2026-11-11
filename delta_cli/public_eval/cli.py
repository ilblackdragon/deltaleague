"""Optional evaluator commands; importing the base CLI never loads benchmark dependencies."""
import json
from pathlib import Path

import typer

app = typer.Typer(help='Run the pinned public benchmark suite against a served checkpoint.')


@app.command('config')
def config_command():
    """Print the public suite, runtime pins and protocol source hashes without a GPU."""
    from .runner import configuration
    typer.echo(json.dumps(configuration(), indent=2))


@app.command('run')
def run_command(endpoint: str = typer.Option(..., help='OpenAI-compatible vLLM endpoint, ending in /v1'),
                model: str = typer.Option('candidate', help='Name of the checkpoint served at the endpoint'),
                tokenizer: Path = typer.Option(..., help='Matching local checkpoint/tokenizer directory'),
                content_hash: str = typer.Option(..., help='Immutable content hash of the served checkpoint'),
                output: Path = typer.Option(Path('public-results'), help='New output directory'),
                manifest: Path | None = typer.Option(None, help='Downloaded pinned validation.json; bundled by default'),
                concurrency: int = typer.Option(32, min=1, max=256),
                evaluation_config: Path | None = typer.Option(None, help="Downloaded evaluation.json with versioned generation settings")):
    """Run all 1,110 public items; save an advisory report, never an official submission."""
    from .runner import evaluate_public
    try:
        config = json.loads(evaluation_config.read_text()) if evaluation_config else None
        if config:
            from ..artifacts import file_hash
            if config.get('protocol_version') != 'delta-public-v1' or not manifest or file_hash(manifest) != config['validation_sha256']:
                raise ValueError('Evaluation configuration and public manifest do not match')
            concurrency = config.get('generation', {}).get('max_num_seqs', concurrency)
        result = evaluate_public(endpoint=endpoint, model=model, tokenizer=tokenizer, content_hash=content_hash,
                                 output=output, manifest_path=manifest, concurrency=concurrency,
                                 **({'generation':config.get('generation', {}).get('public_validation'), 'config_id':config['id']} if config else {}),
                                 progress=lambda done, total: typer.echo(f'Public validation: {done}/{total}', err=True))
    except ImportError:
        typer.echo('Install the pinned public evaluator dependencies in a separate environment; see PUBLIC_EVALUATION.md.', err=True)
        raise typer.Exit(1)
    except Exception as exc:
        typer.echo(f'Public evaluation failed ({type(exc).__name__}); no complete score was accepted. '
                   'Check runtime pins, suite identity and the endpoint. Existing output directories cannot be reused.', err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps({'output': str(output), 'local_only': True,
                          'validation_score': result['validation_score'],
                          'validation_benchmarks': result['validation_benchmarks']}, indent=2))


@app.command('benchmark-config')
def benchmark_config_command(
        benchmark: str = typer.Option(..., help='Native benchmark name, including terminal_bench_2_1'),
        generation: Path = typer.Option(..., help='JSON with explicit generation settings'),
        coverage: str = typer.Option('sample', help='sample or full'),
        concurrency: int = typer.Option(8, min=1, max=256),
        task: list[str] = typer.Option([], help='Native Terminal task ID; repeat for a sample'),
        attempts: int = typer.Option(1, help='Terminal attempts per task: 1 or 5')):
    """Write a standalone protocol to stdout, without loading a model."""
    from .standalone import make_protocol
    try:
        result = make_protocol(benchmark, json.loads(generation.read_text()), coverage=coverage,
                               concurrency=concurrency, task_ids=task, attempts=attempts)
    except (ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2))


@app.command('benchmark')
def benchmark_command(
        protocol: Path = typer.Option(..., help='Protocol from benchmark-config or an exported run'),
        endpoint: str = typer.Option(..., help='Compatible inference endpoint, optionally ending in /v1'),
        tokenizer: Path = typer.Option(..., help='Matching local tokenizer directory'),
        content_hash: str = typer.Option(..., help='Immutable SHA-256 identity of the served model'),
        model: str = typer.Option('candidate', help='Model name accepted by the endpoint'),
        output: Path = typer.Option(Path('benchmark-results'), help='New output directory'),
        start: int | None = typer.Option(None, help='Inclusive shard ordinal'),
        end: int | None = typer.Option(None, help='Exclusive shard ordinal'),
        api_key_env: str | None = typer.Option(None, help='Environment variable containing endpoint credentials')):
    """Run a native benchmark, full set, or shard without the Delta League controller."""
    from .standalone import run_benchmark
    if (start is None) != (end is None):
        raise typer.BadParameter('Supply both --start and --end for a shard')
    try:
        result = run_benchmark(protocol=json.loads(protocol.read_text()), endpoint=endpoint, model=model,
                               tokenizer=tokenizer, content_hash=content_hash, output=output,
                               bounds=None if start is None else [start, end], api_key_env=api_key_env)
    except Exception as exc:
        typer.echo(f'Benchmark failed ({type(exc).__name__}); inspect local evidence. No complete score was accepted.', err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, indent=2))
