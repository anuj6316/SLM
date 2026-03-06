from django.db import models
from django.conf import settings
import uuid

class ScrapeJob(models.Model):
    ## CHOICES 
    STATUS = [
        ("pending", "Pending"),
        ("active", "Active"),
        ("completed", "Completed"),
        ("failed", "Failed")
    ]

    ## 1. Relation to User (Who owns this job?)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete = models.CASCADE,
        related_name = "scrape_jobs"
    )
    
    ## JobId Identification 
    job_id = models.UUIDField(default = uuid.uuid4, unique = True, editable = False)

    ## Input config url, scrape type, depth
    url = models.URLField(max_length=500)
    scrape_type = models.CharField(max_length=50, choices=[('flash', "Flash"), ("deep", "Deep")])
    max_depth = models.IntegerField(default=2)

    ## Progress tracking: status, pages scraped, error message
    status = models.CharField(max_length=50, choices=STATUS, default="pending")
    pages_scraped = models.IntegerField(default=0)
    error_message = models.TextField(blank=True, null=True)

    ## Timestamp 
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.email} - {self.url} ({self.status})"

def document_upload_path(instance, filename):
    # FIX: Uses actual filename variable
    return f"scraped_data/user_{instance.job.user.id}/{instance.job.job_id}/{filename}"

class ScrapedDocument(models.Model):
    ## 1. Relational model
    job = models.ForeignKey(
        'ScrapeJob',
        on_delete = models.CASCADE,
        related_name = "documents"
    )

    page_url = models.URLField(max_length=1000)
    
    ## 2. Result data
    content_file = models.FileField(upload_to=document_upload_path)

    token_count = models.IntegerField(default = 0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Doc from {self.job.job_id} - {self.page_url}"
