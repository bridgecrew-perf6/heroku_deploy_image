

import os
from image_labelling_tool import labelling_tool

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECRET_KEY = '3+p2$qln6o1ws1c)6o!+o+p%ql1n!+tt@wp)g5!pfgliqld)yo'

DEBUG = False
# ALLOWED_HOSTS = []
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
                'django.template.context_processors.static',
            ],
        },
    },
]

WSGI_APPLICATION = 'example_labeller_app.wsgi.application'


# Database
# https://docs.djangoproject.com/en/1.8/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Internationalization
# https://docs.djangoproject.com/en/1.8/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True

WHITENOISE_USE_FINDERS = True

CSRF_TRUSTED_ORIGINS = [
    'https://avistosimage.herokuapp.com'
]

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.8/howto/static-files/

STATIC_URL = '/static/'

STATIC_ROOT = os.path.join(BASE_DIR, 'static')

STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

STATICFILES_DIRS = [os.path.join(BASE_DIR, '/static')]


MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'

CSRF_COOKIE_SECURE = False

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

#
#
# Colour schemes and label classes are stored in the database
#
#


# Annotation controls
# Labels may also have optional meta-data associated with them
# You could use this for e.g. indicating if an object is fully visible, mostly visible or significantly obscured.
# You could also indicate quality (e.g. blurriness, etc)
# There are four types of annotation. They have some common properties:
#   - name: symbolic name (Python identifier)
#   - label_text: label text in UI
#   Check boxes, radio buttons and popup menus also have:
#     - visibility_label_text: [optional] if provided, label visibility can be filtered by this annotation value,
#       in which case a drop down will appear in the UI allowing the user to select a filter value
#       that will hide/show labels accordingly
# Control types:
# Check box (boolean value):
#   `labelling_tool.AnnoControlCheckbox`; only the 3 common parameters listed above
# Radio button (choice from a list):
#   `labelling_tool.AnnoControlRadioButtons`; the 3 common parameters listed above and:
#       choices: list of `labelling_tool.AnnoControlRadioButtons.choice` that provide:
#           value: symbolic value name for choice
#           tooltip: extra information for user
#       label_on_own_line [optional]: if True, place the label and the buttons on a separate line in the UI
# Popup menu (choice from a grouped list):
#   `labelling_tool.AnnoControlPopupMenu`; the 3 common parameters listed above and:
#       groups: list of groups `labelling_tool.AnnoControlPopupMenu.group`:
#           label_text: label text in UI
#           choices: list of `labelling_tool.AnnoControlPopupMenu.choice` that provide:
#               value: symbolic value name for choice
#               label_text: choice label text in UI
#               tooltip: extra information for user
# Text (free form plain text):
#   `labelling_tool.AnnoControlText`; only the 2 common parameters listed above and:
#       - multiline: boolean; if True a text area will be used, if False a single line text entry
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

