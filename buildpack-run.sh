cd engine
python setup.py install
cd ..
export FLASK_APP=app
export FLASK_ENV=development
flask run