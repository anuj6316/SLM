from django.contrib import admin
from .models import ScrapeJob, ScrapedDocument

# Register your models here.

## 1. Create the inline for documents
class ScrapedDocumentInline(admin.TabularInline):
    """
    Shows the scraped pages directly inside the job detail page.
    """
    model = ScrapedDocument
    extra = 0 ## Prevents Django from showing 3 empty rows by default
    # Nobody should edit the file path manually in admin
    readonly_fields = ('page_url', 'content_file', 'token_count', 'created_at')
    can_delete = False


## 2. Register the main ScrapeJob
@admin.register(ScrapeJob)
class ScrapeJobAdmin(admin.ModelAdmin):
    ## What columns shows up in the main list
    list_display = ('job_id', 'user', 'url', 'scrape_type', 'status', 'pages_scraped', 'created_at')

    ## what can we filter by on the right side?
    list_filter = ('status', 'scrape_type', 'created_at')

    ## what can we search for in the top search bar?
    ## Note the double underscore 'user__email' to search inside th foriegnKey
    search_fields = ('job_id', 'user__email', 'url')

    ## what fields should be locked?
    readonly_fields = ('job_id', 'created_at', 'updated_at', 'pages_scraped')

    ## attach the inline we made above
    inlines = [ScrapedDocumentInline]

## 3. Register the document model seprately (optional but good)
@admin.register(ScrapedDocument)
class ScrapedDocumentAdmin(admin.ModelAdmin):
    list_display = ('page_url', 'job', 'token_count', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('page_url', 'job__job_id')
    readonly_fields = ('created_at',)