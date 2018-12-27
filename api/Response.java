package api;

/**
 * Data class to hold result of Search. status field tells whether the result of
 * the search was "Success" or "Failure". Class cannot be explictly be created
 * without the ResponseParser class. Only gettter methods public.
 * 
 * @see ResponseParser
 */
public class Response {
    private String detailedDescription;
    private String descriptionUrl;
    private String imageUrl;
    private String contentUrl;
    private String status;

    Response() {
        status = "Failure";
    }

    public String getDetailedDescription() {
        return detailedDescription;
    }

    void setDetailedDescription(String detailedDescription) {
        this.detailedDescription = detailedDescription;
    }

    public String getDescriptionUrl() {
        return descriptionUrl;
    }

    void setDescriptionUrl(String descriptionUrl) {
        this.descriptionUrl = descriptionUrl;
    }

    public String getImageUrl() {
        return imageUrl;
    }

    void setImageUrl(String imageUrl) {
        this.imageUrl = imageUrl;
    }

    public String getContentUrl() {
        return contentUrl;
    }

    void setContentUrl(String contentUrl) {
        this.contentUrl = contentUrl;
    }

    public String getStatus() {
        return status;
    }

    void setStatus(String status) {
        this.status = status;
    }
}
