MAIN = src/main.py
MAIN_TEST = test/main_test.py

default: run

run: 
	python3 $(MAIN)

install: requirements.txt
	pip3 install -r requirements.txt

test:
	python3 $(MAIN_TEST)

clean:
	rm -rf __pycache__ test_output/*
