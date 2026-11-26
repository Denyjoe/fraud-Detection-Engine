import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Minimal client for the moderation engine's POST /moderate endpoint (Java 11+, no dependencies).
 *
 * <p>Usage:
 * <pre>
 *   javac ModerationClient.java
 *   java ModerationClient "Raba nzuri za Nike" "Size 42" "C:\\Photos\\raba 1.jpg"
 * </pre>
 *
 * <p>Act on the "decision" field of the JSON response:
 * ALLOW -> publish; REVIEW -> hold for a moderator; BLOCK -> reject and show "seller_message_sw";
 * ESCALATE -> hide immediately and alert the safety team (urgent human review).
 * Parse the JSON with your usual library (Jackson, Gson ...).
 */
public final class ModerationClient {

    // HTTP/1.1: uvicorn rejects the HTTP/2 cleartext upgrade (h2c) that Java attempts by default.
    private final HttpClient http = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(Duration.ofSeconds(10))
            .build();
    private final String baseUrl;
    private final String apiKey; // value of MODERATION_API_KEY on the server, or null

    public ModerationClient(String baseUrl, String apiKey) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.apiKey = apiKey;
    }

    /** Sends a listing (text fields + any number of image/video files) and returns the JSON response. */
    public String moderate(Map<String, String> fields, List<Path> files) throws IOException, InterruptedException {
        String boundary = "----moderation" + UUID.randomUUID();
        ByteArrayOutputStream body = new ByteArrayOutputStream();
        for (Map.Entry<String, String> field : fields.entrySet()) {
            write(body, "--" + boundary + "\r\n");
            write(body, "Content-Disposition: form-data; name=\"" + field.getKey() + "\"\r\n");
            write(body, "Content-Type: text/plain; charset=UTF-8\r\n\r\n");
            write(body, field.getValue() + "\r\n");
        }
        for (Path file : files) {
            String type = Files.probeContentType(file);
            write(body, "--" + boundary + "\r\n");
            write(body, "Content-Disposition: form-data; name=\"files\"; filename=\""
                    + file.getFileName().toString().replace("\"", "") + "\"\r\n");
            write(body, "Content-Type: " + (type != null ? type : "application/octet-stream") + "\r\n\r\n");
            body.write(Files.readAllBytes(file));
            write(body, "\r\n");
        }
        write(body, "--" + boundary + "--\r\n");

        HttpRequest.Builder request = HttpRequest.newBuilder(URI.create(baseUrl + "/moderate"))
                .timeout(Duration.ofSeconds(120)) // first call loads the models; video takes longer
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .POST(HttpRequest.BodyPublishers.ofByteArray(body.toByteArray()));
        if (apiKey != null && !apiKey.isEmpty()) {
            request.header("X-API-Key", apiKey);
        }
        HttpResponse<String> response = http.send(request.build(), HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        if (response.statusCode() != 200) {
            // 401 bad key, 413 too many/too large files, 422 invalid input, 500 retry later -> treat as REVIEW
            throw new IOException("Moderation failed: HTTP " + response.statusCode() + " " + response.body());
        }
        return response.body();
    }

    /** True when the moderation service and its models are reachable. */
    public boolean healthy() throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/health")).GET().build();
        return http.send(request, HttpResponse.BodyHandlers.ofString()).statusCode() == 200;
    }

    private static void write(ByteArrayOutputStream out, String text) throws IOException {
        out.write(text.getBytes(StandardCharsets.UTF_8));
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: java ModerationClient <title> [description] [file ...]");
            System.exit(2);
        }
        ModerationClient client = new ModerationClient(
                System.getenv().getOrDefault("MODERATION_URL", "http://127.0.0.1:8000"),
                System.getenv("MODERATION_API_KEY"));
        System.out.println("healthy: " + client.healthy());
        Map<String, String> fields = Map.of(
                "title", args[0],
                "description", args.length > 1 ? args[1] : "",
                "listing_id", "LIST-" + UUID.randomUUID().toString().substring(0, 8),
                "seller_id", "SELLER-123");
        List<Path> files = new java.util.ArrayList<>();
        for (int i = 2; i < args.length; i++) {
            files.add(Path.of(args[i]));
        }
        System.out.println(client.moderate(fields, files));
    }
}
