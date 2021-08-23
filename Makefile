docs: 
	pdoc --output-dir=docs --force rpanda
	mv docs/rpanda/* docs/
	rm -rf docs/rpanda