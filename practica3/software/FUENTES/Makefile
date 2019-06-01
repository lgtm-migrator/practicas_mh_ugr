ALGORITHMS := ils anneal 'de/rand/one' 'de/current-to-best/one'
DATESETS := coloposcopy ionosphere texture

install:
	echo "Creating virtual environment..."
	python3 -m venv env
	echo "Installing dependencies..."
	./env/bin/pip3 install -r requirements.txt

clean:
	rm -rf env


$(ALGORITHMS):
	python3 practica3.py colposcopy $@ --trace --to_excel
	python3 practica3.py ionosphere $@ --trace --to_excel
	python3 practica3.py texture $@ --trace --to_excel


run_all: $(ALGORITHMS)
