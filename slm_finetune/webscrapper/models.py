from django.db import models

# Create your models here.
class ScrapeStatus(models.Model):
    job_id = models.CharField(max_length=255)
    url = models.URLField()
    scrape_type = models.CharField(max_length=30)
    status = models.CharField(max_length=255)
    number_of_links_scrapped = models.IntegerField()

    created_at = models.DateTimeField(auto_now_add=True)
    udpated_at = models.DateTimeField(auto_now=True)

    files = models.FileField("documents/")
    message = models.TextField()

    def __str__(self):
        return self.job_id