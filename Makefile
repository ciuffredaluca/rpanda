clean-docs:
	rm -rf docs

docs: clean-docs
	pdoc --html -o ./docs rpanda
	mv docs/rpanda/* docs/
	rm -rf docs/rpanda