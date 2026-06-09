from django.urls import path
from . import views

urlpatterns = [
    path('',              views.dashboard,           name='dashboard'),
    path('incidents/',    views.incidents,            name='incidents'),
    path('analytics/',    views.analytics,            name='analytics'),
    path('settings/',     views.settings_view,        name='settings'),
    path('devices/',      views.devices,              name='devices'),
    path('video_feed/',   views.video_feed,           name='video_feed'),
    path('api/latest_detection/', views.latest_detection_api, name='latest_detection'),
    path('incidents/clear/', views.clear_incidents,  name='clear_incidents'),
]
