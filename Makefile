install:
	pip install --upgrade pip
	pip install matplotlib
	pip install tensorflow
	pip install numpy
	apt-get install python-tk -y

run:
	python test.py
