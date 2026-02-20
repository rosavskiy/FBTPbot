/*
 * Интеграция Java-приложения Фармбазис с AI-техподдержкой
 * 
 * API endpoint: http://41.216.182.31:8000/api/chat
 * Требуется: Java 11+ (для java.net.http.HttpClient)
 */

// ═══════════════════════════════════════════════════════════════
// 1. ПРОСТОЙ КЛИЕНТ (Java 11+ HttpClient)
// ═══════════════════════════════════════════════════════════════

package ru.farmbazis.support;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonArray;

public class FarmbazisAIClient {
    private static final String API_URL = "http://41.216.182.31:8000/api/chat";
    private static final Duration TIMEOUT = Duration.ofSeconds(30);
    
    private final HttpClient httpClient;
    private final Gson gson;
    private String sessionId;
    
    public FarmbazisAIClient() {
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(TIMEOUT)
            .build();
        this.gson = new Gson();
        this.sessionId = null;
    }
    
    /**
     * Отправить вопрос боту
     * 
     * @param question текст вопроса
     * @return ответ бота
     * @throws Exception если ошибка соединения
     */
    public BotResponse ask(String question) throws Exception {
        // Формируем JSON запрос
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("message", question);
        if (sessionId != null) {
            requestBody.addProperty("session_id", sessionId);
        }
        
        // HTTP запрос
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(API_URL))
            .timeout(TIMEOUT)
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(requestBody)))
            .build();
        
        // Отправляем и получаем ответ
        HttpResponse<String> response = httpClient.send(
            request, 
            HttpResponse.BodyHandlers.ofString()
        );
        
        if (response.statusCode() != 200) {
            throw new Exception("HTTP " + response.statusCode() + ": " + response.body());
        }
        
        // Парсим JSON
        JsonObject data = gson.fromJson(response.body(), JsonObject.class);
        
        // Сохраняем session_id для следующих запросов
        this.sessionId = data.get("session_id").getAsString();
        
        return parseBotResponse(data);
    }
    
    /**
     * Сбросить контекст диалога
     */
    public void resetSession() {
        this.sessionId = null;
    }
    
    private BotResponse parseBotResponse(JsonObject data) {
        BotResponse response = new BotResponse();
        response.type = data.get("response_type").getAsString();
        response.text = data.get("answer").getAsString();
        response.confidence = data.get("confidence").getAsDouble();
        response.needsEscalation = data.get("needs_escalation").getAsBoolean();
        
        // YouTube ссылки
        if (data.has("youtube_links")) {
            JsonArray links = data.getAsJsonArray("youtube_links");
            response.youtubeLinks = new String[links.size()];
            for (int i = 0; i < links.size(); i++) {
                response.youtubeLinks[i] = links.get(i).getAsString();
            }
        }
        
        // Уточняющие темы
        if ("clarification".equals(response.type) && data.has("suggested_topics")) {
            JsonArray topics = data.getAsJsonArray("suggested_topics");
            response.topics = new Topic[topics.size()];
            for (int i = 0; i < topics.size(); i++) {
                JsonObject t = topics.get(i).getAsJsonObject();
                Topic topic = new Topic();
                topic.number = i + 1;
                topic.title = t.get("title").getAsString();
                topic.articleId = t.get("article_id").getAsString();
                topic.snippet = t.get("snippet").getAsString();
                response.topics[i] = topic;
            }
        }
        
        return response;
    }
}

// ═══════════════════════════════════════════════════════════════
// 2. МОДЕЛИ ДАННЫХ
// ═══════════════════════════════════════════════════════════════

class BotResponse {
    public String type;           // "answer" или "clarification"
    public String text;           // текст ответа
    public double confidence;     // уверенность 0.0-1.0
    public boolean needsEscalation; // нужен оператор
    public String[] youtubeLinks; // видео-инструкции
    public Topic[] topics;        // темы для уточнения (если type="clarification")
    
    public boolean isAnswer() {
        return "answer".equals(type);
    }
    
    public boolean isClarification() {
        return "clarification".equals(type);
    }
}

class Topic {
    public int number;       // номер темы (1, 2, 3...)
    public String title;     // заголовок
    public String articleId; // ID статьи в БЗ
    public String snippet;   // краткий фрагмент
}

// ═══════════════════════════════════════════════════════════════
// 3. ПРИМЕР ИСПОЛЬЗОВАНИЯ В SWING UI
// ═══════════════════════════════════════════════════════════════

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class SupportDialog extends JDialog {
    private FarmbazisAIClient aiClient;
    private JTextArea questionArea;
    private JTextArea answerArea;
    private JPanel topicsPanel;
    private JButton askButton;
    private JButton resetButton;
    
    public SupportDialog(Frame owner) {
        super(owner, "Техподдержка AI", true);
        this.aiClient = new FarmbazisAIClient();
        
        initUI();
        setSize(800, 600);
        setLocationRelativeTo(owner);
    }
    
    private void initUI() {
        setLayout(new BorderLayout(10, 10));
        
        // Поле вопроса
        JPanel questionPanel = new JPanel(new BorderLayout(5, 5));
        questionPanel.setBorder(BorderFactory.createTitledBorder("Ваш вопрос"));
        questionArea = new JTextArea(3, 40);
        questionArea.setLineWrap(true);
        questionPanel.add(new JScrollPane(questionArea), BorderLayout.CENTER);
        
        JPanel buttonsPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        askButton = new JButton("Спросить");
        askButton.addActionListener(e -> handleAsk());
        resetButton = new JButton("Новый диалог");
        resetButton.addActionListener(e -> handleReset());
        buttonsPanel.add(askButton);
        buttonsPanel.add(resetButton);
        questionPanel.add(buttonsPanel, BorderLayout.SOUTH);
        
        add(questionPanel, BorderLayout.NORTH);
        
        // Панель для уточняющих тем (изначально скрыта)
        topicsPanel = new JPanel();
        topicsPanel.setLayout(new BoxLayout(topicsPanel, BoxLayout.Y_AXIS));
        topicsPanel.setBorder(BorderFactory.createTitledBorder("Выберите подходящую тему"));
        topicsPanel.setVisible(false);
        add(topicsPanel, BorderLayout.CENTER);
        
        // Поле ответа
        JPanel answerPanel = new JPanel(new BorderLayout(5, 5));
        answerPanel.setBorder(BorderFactory.createTitledBorder("Ответ"));
        answerArea = new JTextArea(15, 40);
        answerArea.setLineWrap(true);
        answerArea.setWrapStyleWord(true);
        answerArea.setEditable(false);
        answerPanel.add(new JScrollPane(answerArea), BorderLayout.CENTER);
        add(answerPanel, BorderLayout.SOUTH);
    }
    
    private void handleAsk() {
        String question = questionArea.getText().trim();
        if (question.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Введите вопрос", "Ошибка", JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        askButton.setEnabled(false);
        answerArea.setText("Обработка запроса...");
        
        // Выполняем в фоновом потоке
        SwingWorker<BotResponse, Void> worker = new SwingWorker<>() {
            @Override
            protected BotResponse doInBackground() throws Exception {
                return aiClient.ask(question);
            }
            
            @Override
            protected void done() {
                try {
                    BotResponse response = get();
                    handleResponse(response);
                } catch (Exception e) {
                    answerArea.setText("Ошибка: " + e.getMessage());
                } finally {
                    askButton.setEnabled(true);
                }
            }
        };
        worker.execute();
    }
    
    private void handleResponse(BotResponse response) {
        if (response.isClarification()) {
            // Показываем темы для выбора
            topicsPanel.removeAll();
            topicsPanel.setVisible(true);
            
            answerArea.setText(response.text);
            
            ButtonGroup group = new ButtonGroup();
            for (Topic topic : response.topics) {
                JRadioButton radio = new JRadioButton(
                    String.format("%d. %s", topic.number, topic.title)
                );
                radio.addActionListener(e -> {
                    // Пользователь выбрал тему
                    questionArea.setText(String.valueOf(topic.number));
                    handleAsk();
                });
                group.add(radio);
                topicsPanel.add(radio);
            }
            
            topicsPanel.revalidate();
            topicsPanel.repaint();
            
        } else {
            // Показываем ответ
            topicsPanel.setVisible(false);
            
            StringBuilder answer = new StringBuilder();
            answer.append(response.text);
            
            if (response.youtubeLinks != null && response.youtubeLinks.length > 0) {
                answer.append("\n\n📹 Видео-инструкции:\n");
                for (String link : response.youtubeLinks) {
                    answer.append("  • ").append(link).append("\n");
                }
            }
            
            if (response.needsEscalation) {
                answer.append("\n\n⚠️ Рекомендуется обратиться к оператору техподдержки");
            }
            
            answer.append(String.format("\n\nУверенность: %.0f%%", response.confidence * 100));
            
            answerArea.setText(answer.toString());
        }
    }
    
    private void handleReset() {
        aiClient.resetSession();
        questionArea.setText("");
        answerArea.setText("");
        topicsPanel.setVisible(false);
        topicsPanel.removeAll();
    }
}

// ═══════════════════════════════════════════════════════════════
// 4. ИНТЕГРАЦИЯ В ГЛАВНОЕ ОКНО ФАРМБАЗИС
// ═══════════════════════════════════════════════════════════════

public class MainWindow extends JFrame {
    private FarmbazisAIClient aiClient;
    
    public MainWindow() {
        this.aiClient = new FarmbazisAIClient();
        
        // Кнопка справки в меню
        JMenu helpMenu = new JMenu("Справка");
        JMenuItem aiHelpItem = new JMenuItem("AI-Помощник");
        aiHelpItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F1, 0));
        aiHelpItem.addActionListener(e -> showAIHelp());
        helpMenu.add(aiHelpItem);
        
        JMenuBar menuBar = new JMenuBar();
        menuBar.add(helpMenu);
        setJMenuBar(menuBar);
    }
    
    private void showAIHelp() {
        SupportDialog dialog = new SupportDialog(this);
        dialog.setVisible(true);
    }
    
    /**
     * Контекстная помощь — задать вопрос по текущему окну
     */
    public void showContextHelp(String context) {
        try {
            String question = "Как работать с " + context + "?";
            BotResponse response = aiClient.ask(question);
            
            if (response.isAnswer()) {
                JOptionPane.showMessageDialog(
                    this,
                    response.text,
                    "Помощь: " + context,
                    JOptionPane.INFORMATION_MESSAGE
                );
            } else {
                // Открываем полный диалог для уточнения
                showAIHelp();
            }
        } catch (Exception e) {
            JOptionPane.showMessageDialog(
                this,
                "Ошибка подключения к серверу помощи: " + e.getMessage(),
                "Ошибка",
                JOptionPane.ERROR_MESSAGE
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// 5. АЛЬТЕРНАТИВА: ИСПОЛЬЗОВАНИЕ OkHttp (более популярная либа)
// ═══════════════════════════════════════════════════════════════

/*
 * Добавьте в pom.xml:
 * <dependency>
 *   <groupId>com.squareup.okhttp3</groupId>
 *   <artifactId>okhttp</artifactId>
 *   <version>4.12.0</version>
 * </dependency>
 */

import okhttp3.*;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class FarmbazisAIClientOkHttp {
    private static final String API_URL = "http://41.216.182.31:8000/api/chat";
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    
    private final OkHttpClient client;
    private final Gson gson;
    private String sessionId;
    
    public FarmbazisAIClientOkHttp() {
        this.client = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .build();
        this.gson = new Gson();
    }
    
    public BotResponse ask(String question) throws IOException {
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("message", question);
        if (sessionId != null) {
            requestBody.addProperty("session_id", sessionId);
        }
        
        RequestBody body = RequestBody.create(gson.toJson(requestBody), JSON);
        Request request = new Request.Builder()
            .url(API_URL)
            .post(body)
            .build();
        
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("HTTP " + response.code() + ": " + response.message());
            }
            
            JsonObject data = gson.fromJson(response.body().string(), JsonObject.class);
            this.sessionId = data.get("session_id").getAsString();
            
            return parseBotResponse(data);
        }
    }
    
    // parseBotResponse() - такой же как выше
}

// ═══════════════════════════════════════════════════════════════
// 6. ПРИМЕР ПРОСТОГО ТЕКСТОВОГО КЛИЕНТА (консоль)
// ═══════════════════════════════════════════════════════════════

import java.util.Scanner;

public class ConsoleSupportClient {
    public static void main(String[] args) {
        FarmbazisAIClient client = new FarmbazisAIClient();
        Scanner scanner = new Scanner(System.in);
        
        System.out.println("=== Фармбазис AI-Техподдержка ===");
        System.out.println("Введите 'exit' для выхода, 'reset' для нового диалога\n");
        
        while (true) {
            System.out.print("Вопрос: ");
            String input = scanner.nextLine().trim();
            
            if (input.equalsIgnoreCase("exit")) {
                break;
            }
            
            if (input.equalsIgnoreCase("reset")) {
                client.resetSession();
                System.out.println("Контекст диалога сброшен.\n");
                continue;
            }
            
            if (input.isEmpty()) {
                continue;
            }
            
            try {
                BotResponse response = client.ask(input);
                
                if (response.isClarification()) {
                    System.out.println("\n" + response.text);
                    System.out.println("\nВыберите подходящую тему:");
                    for (Topic topic : response.topics) {
                        System.out.printf("  %d. %s\n", topic.number, topic.title);
                    }
                    System.out.println("\nВведите номер темы или опишите подробнее:");
                    
                } else {
                    System.out.println("\nОтвет (уверенность " + 
                        String.format("%.0f%%", response.confidence * 100) + "):");
                    System.out.println(response.text);
                    
                    if (response.youtubeLinks != null && response.youtubeLinks.length > 0) {
                        System.out.println("\n📹 Видео-инструкции:");
                        for (String link : response.youtubeLinks) {
                            System.out.println("  " + link);
                        }
                    }
                    
                    if (response.needsEscalation) {
                        System.out.println("\n⚠️ Рекомендуется обратиться к оператору");
                    }
                }
                
                System.out.println();
                
            } catch (Exception e) {
                System.err.println("Ошибка: " + e.getMessage());
            }
        }
        
        scanner.close();
        System.out.println("До свидания!");
    }
}

// ═══════════════════════════════════════════════════════════════
// 7. Maven dependencies (pom.xml)
// ═══════════════════════════════════════════════════════════════

/*
<dependencies>
    <!-- JSON парсинг -->
    <dependency>
        <groupId>com.google.code.gson</groupId>
        <artifactId>gson</artifactId>
        <version>2.10.1</version>
    </dependency>
    
    <!-- HTTP клиент (опционально, если не используете Java 11+ HttpClient) -->
    <dependency>
        <groupId>com.squareup.okhttp3</groupId>
        <artifactId>okhttp</artifactId>
        <version>4.12.0</version>
    </dependency>
</dependencies>
*/

// ═══════════════════════════════════════════════════════════════
// 8. ОБРАБОТКА ОШИБОК В PRODUCTION
// ═══════════════════════════════════════════════════════════════

class AIException extends Exception {
    private final int httpCode;
    
    public AIException(String message, int httpCode) {
        super(message);
        this.httpCode = httpCode;
    }
    
    public boolean isServerError() {
        return httpCode >= 500;
    }
    
    public boolean isClientError() {
        return httpCode >= 400 && httpCode < 500;
    }
    
    public boolean shouldRetry() {
        return isServerError() || httpCode == 429; // 429 = Too Many Requests
    }
}

// Использование:
public BotResponse askWithRetry(String question, int maxRetries) throws AIException {
    int attempts = 0;
    Exception lastException = null;
    
    while (attempts < maxRetries) {
        try {
            return ask(question);
        } catch (Exception e) {
            lastException = e;
            attempts++;
            
            if (attempts < maxRetries) {
                try {
                    Thread.sleep(1000 * attempts); // экспоненциальная задержка
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }
    }
    
    throw new AIException("Не удалось получить ответ после " + maxRetries + " попыток", 0);
}
