ALGORITHMS := relief local-search agg-blx agg-ca age-blx age-ca 'AM-(1,1.0)' 'AM-(1,0.1)' 'AM-(1,0.1mej)'
DATESETS := coloposcopy ionosphere texture
PYTHON := ./env/bin/python3


install:
	echo "Creating virtual environment..."
	python3 -m venv env
	echo "Installing dependencies..."
	./env/bin/pip3 install -r requirements.txt

clean:
	rm -rf env


$(ALGORITHMS):
	python3 practica2.py colposcopy $@ --trace --to_excel
	python3 practica2.py ionosphere $@ --trace --to_excel
	python3 practica2.py texture $@ --trace --to_excel


run_all: $(ALGORITHMS)
