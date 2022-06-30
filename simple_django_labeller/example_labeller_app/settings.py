from pathlib import Path
import os
from decouple import config
from image_labelling_tool import labelling_tool

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-@@7uqw0%@1z_!$r!t+-8r96ujsa0*prkeo086tl5@6mb5@dge1'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

ALLOWED_HOSTS = ['.herokuapp.com']


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'image_labelling_tool.apps.ImageLabellingToolConfig',
    'example_labeller.apps.ExampleLabellerConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'user_visit.middleware.UserVisitMiddleware',
]

ROOT_URLCONF = 'example_labeller_app.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'example_labeller_app.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Internationalization
# https://docs.djangoproject.com/en/4.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'Asia/Kolkata'

USE_I18N = True

USE_TZ = True

WHITENOISE_USE_FINDERS = True

CSRF_TRUSTED_ORIGINS = [
    'https://image-avistos.herokuapp.com'
]

STATIC_URL = 'static/'

STATICFILES_DIRS = [
    os.path.join(BASE_DIR,'./image_labelling_tool/static')
]

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


STATIC_ROOT = os.path.join(BASE_DIR, 'static')

STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

DEFAULT_FILE_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'



#-------------------------------------------------------------------------------------------
ANNO_CONTROLS = [
    labelling_tool.AnnoControlCheckbox('good_quality', 'Good quality',
                                       visibility_label_text='Filter by good quality'),
    labelling_tool.AnnoControlRadioButtons('visibility', 'Visible', choices=[
        labelling_tool.AnnoControlRadioButtons.choice(value='full', label_text='Fully',
                                                      tooltip='Object is fully visible'),
        labelling_tool.AnnoControlRadioButtons.choice(value='mostly', label_text='Mostly',
                                                      tooltip='Object is mostly visible'),
        labelling_tool.AnnoControlRadioButtons.choice(value='obscured', label_text='Obscured',
                                                      tooltip='Object is significantly obscured'),
    ], label_on_own_line=False, visibility_label_text='Filter by visibility'),
    labelling_tool.AnnoControlPopupMenu('material', 'Material', groups=[
        labelling_tool.AnnoControlPopupMenu.group(label_text='Artifical/buildings', choices=[
            labelling_tool.AnnoControlPopupMenu.choice(value='concrete', label_text='Concrete',
                                                       tooltip='Concrete objects'),
            labelling_tool.AnnoControlPopupMenu.choice(value='plastic', label_text='Plastic',
                                                       tooltip='Plastic objects'),
            labelling_tool.AnnoControlPopupMenu.choice(value='asphalt', label_text='Asphalt',
                                                       tooltip='Road, pavement, etc.'),
        ]),
        labelling_tool.AnnoControlPopupMenu.group(label_text='Flat natural', choices=[
            labelling_tool.AnnoControlPopupMenu.choice(value='grass', label_text='Grass',
                                                       tooltip='Grass covered ground'),
            labelling_tool.AnnoControlPopupMenu.choice(value='water', label_text='Water', tooltip='Water/lake')]),
        labelling_tool.AnnoControlPopupMenu.group(label_text='Vegetation', choices=[
            labelling_tool.AnnoControlPopupMenu.choice(value='trees', label_text='Trees', tooltip='Trees'),
            labelling_tool.AnnoControlPopupMenu.choice(value='shrubbery', label_text='Shrubs',
                                                       tooltip='Shrubs/bushes'),
            labelling_tool.AnnoControlPopupMenu.choice(value='flowers', label_text='Flowers',
                                                       tooltip='Flowers'),
            labelling_tool.AnnoControlPopupMenu.choice(value='ivy', label_text='Ivy', tooltip='Ivy')]),
    ], visibility_label_text='Filter by material'),
    # labelling_tool.AnnoControlText('comment', 'Comment', multiline=False),
]

# Configuration
LABELLING_TOOL_CONFIG = {
    'useClassSelectorPopup': True,
    'tools': {
        'imageSelector': True,
        'labelClassSelector': True,
        'brushSelect': True,
        'labelClassFilter': True,
        'drawPointLabel': False,
        'drawBoxLabel': True,
        'drawOrientedEllipseLabel': True,
        'drawPolyLabel': True,
        'deleteLabel': True,
        'deleteConfig': {
            'typePermissions': {
                'point': True,
                'box': True,
                'polygon': True,
                'composite': True,
                'group': True,
            }
        }
    },
    'settings': {
        'brushWheelRate': 0.025,  # Change rate for brush radius (mouse wheel)
        'brushKeyRate': 2.0,    # Change rate for brush radius (keyboard)
    }
}

LABELLING_TOOL_ENABLE_LOCKING = False
LABELLING_TOOL_DEXTR_AVAILABLE = False
LABELLING_TOOL_DEXTR_POLLING_INTERVAL = 1000
LABELLING_TOOL_DEXTR_WEIGHTS_PATH = None


LABELLING_TOOL_EXTERNAL_LABEL_API = False
LABELLING_TOOL_EXTERNAL_LABEL_API_URL = 'http://localhost:3000/get_labels'


CELERY_BROKER_URL = 'amqp://guest@localhost//'
CELERY_RESULT_BACKEND = 'rpc://'

CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'

