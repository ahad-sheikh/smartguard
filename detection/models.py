from django.db import models
from django.utils import timezone

class Incident(models.Model):
    WEAPON_TYPES = [
        ('gun',   'Gun'),
        ('knife', 'Knife'),
        ('sword', 'Sword'),
    ]
    DETECTION_TYPES = [
        ('weapon', 'Weapon Only'),
        ('armed',  'Armed Person'),
    ]

    weapon_type    = models.CharField(max_length=20, choices=WEAPON_TYPES)
    detection_type = models.CharField(max_length=20, choices=DETECTION_TYPES, default='weapon')
    confidence     = models.FloatField()
    camera_source  = models.CharField(max_length=100, default='Webcam 01')
    timestamp      = models.DateTimeField(default=timezone.now)
    frame_path     = models.CharField(max_length=500, blank=True, null=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.weapon_type} ({self.confidence:.1f}%) @ {self.timestamp:%H:%M:%S}"


class SystemSettings(models.Model):
    camera_source  = models.CharField(max_length=20, default='webcam')
    iphone_url     = models.CharField(max_length=200, default='http://192.168.0.102:8081/video')
    resolution     = models.CharField(max_length=20, default='320x240')
    confidence     = models.FloatField(default=50)
    frame_skip     = models.IntegerField(default=3)
    detect_gun     = models.BooleanField(default=True)
    detect_knife   = models.BooleanField(default=True)
    detect_sword   = models.BooleanField(default=True)
    armed_person   = models.BooleanField(default=True)

    class Meta:
        verbose_name = 'System Settings'

    def __str__(self):
        return 'System Settings'

    @classmethod
    def get(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class SystemLog(models.Model):
    LEVELS = [('info', 'Info'), ('warn', 'Warning'), ('error', 'Error')]
    message   = models.TextField()
    level     = models.CharField(max_length=10, choices=LEVELS, default='info')
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"[{self.level}] {self.message}"
