MAIN = src/main.py

default: run

run: 
	python3 $(MAIN)

install: requirements.txt
	pip3 install -r requirements.txt

clean:
	rm -rf __pycache__
