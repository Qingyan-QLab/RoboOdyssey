#!/usr/bin/env python3
"""JSONL entry point for the public-observation API agent (separate host Python)."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.agents.openai_api import OpenAIAPIAgent, load_config
from benchmark.agents.stream import serve


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='API agent JSON, not an experiment config')
    parser.add_argument('--check-config', action='store_true', help='validate only; no credential read, API call or output creation')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (ValueError, OSError):
        parser.error('invalid API agent config; verify model ID, fields, URL, budgets and log_dir')
    if args.check_config:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    try:
        agent = OpenAIAPIAgent(config)
    except (ValueError, OSError):
        parser.error('cannot initialize API agent; check credential environment variable and output directory')
    print('API agent artifacts: ' + str(agent.log_dir), file=sys.stderr, flush=True)
    try:
        serve(agent.handle, history_frames=0)
    finally:
        agent.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
