cd engine
python setup.py install
cd ..
$env:FLASK_APP="app"
$env:FLASK_ENV="development"
flask run