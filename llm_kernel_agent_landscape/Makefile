.PHONY: validate mermaid sync-readme render svg pptx drawio all clean

validate:
	python scripts/validate_entries.py

mermaid:
	python scripts/render_mermaid.py

sync-readme: mermaid
	python scripts/sync_readme.py

render: mermaid

svg:
	python scripts/render_landscape.py --view report
	python scripts/render_landscape.py --view full

pptx:
	python scripts/render_pptx.py

drawio:
	python scripts/render_drawio.py

all: validate sync-readme

clean:
	rm -f figures/*.svg figures/*.png figures/*.pdf figures/*.pptx figures/*.drawio figures/*.mmd
