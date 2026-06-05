.PHONY: compile dry-run validate clean

compile:
	find . -name '*.py' -print0 | xargs -0 python -m py_compile

dry-run:
	bash scripts/dry_run_configs.sh

validate: compile dry-run

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf outputs

