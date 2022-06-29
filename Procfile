web: gunicorn example_labeller_app.wsgi
release: python manage.py migrate
release:python simple_django_labeller/manage.py import_schema default demo
release:python simple_django_labeller/manage.py populate media
release:python simple_django_labeller/manage.py runserver