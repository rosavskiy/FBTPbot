# -*- coding: utf-8 -*-
"""
Примеры интеграции ПО Фармбазис с AI-техподдержкой через HTTP API.
"""

import requests
import json

API_URL = "http://41.216.182.31:8000/api/chat"

# ═══════════════════════════════════════════════════════════════
# Пример 1: Простой вопрос-ответ
# ═══════════════════════════════════════════════════════════════

def simple_question():
    """Отправка простого вопроса."""
    payload = {
        "message": "Как сделать возврат товара?",
        "session_id": "user_12345"  # ID пользователя/сессии из вашего ПО
    }
    
    response = requests.post(API_URL, json=payload, timeout=30)
    data = response.json()
    
    if data["response_type"] == "answer":
        print(f"Ответ: {data['answer']}")
        print(f"Уверенность: {data['confidence']:.0%}")
        print(f"Уровень: {data['confidence_level']}")
        print(f"Описание: {data['confidence_label']}")
        
        if data["youtube_links"]:
            print("Видео-инструкции:")
            for link in data["youtube_links"]:
                print(f"  - {link}")
        
        if data["source_articles"]:
            print(f"Источники: статьи {', '.join(data['source_articles'])}")


# ═══════════════════════════════════════════════════════════════
# Пример 2: Уточняющие вопросы
# ═══════════════════════════════════════════════════════════════

def clarification_flow():
    """Обработка уточняющих вопросов."""
    session_id = "user_67890"
    
    # Шаг 1: Размытый вопрос
    payload = {
        "message": "Проблема с накладной",
        "session_id": session_id
    }
    
    response = requests.post(API_URL, json=payload, timeout=30)
    data = response.json()
    
    if data["response_type"] == "clarification":
        print("Бот просит уточнить:")
        print(data["answer"])
        print("\nВарианты:")
        for i, topic in enumerate(data["suggested_topics"], 1):
            print(f"  {i}. {topic['title']}")
        
        # Шаг 2: Пользователь выбрал тему (например, 1)
        choice_payload = {
            "message": "1",  # Номер выбранной темы
            "session_id": session_id  # ТОТ ЖЕ session_id!
        }
        
        response2 = requests.post(API_URL, json=choice_payload, timeout=30)
        data2 = response2.json()
        
        print(f"\nОтвет после уточнения: {data2['answer']}")


# ═══════════════════════════════════════════════════════════════
# Пример 3: Многошаговый диалог с историей
# ═══════════════════════════════════════════════════════════════

def conversation():
    """Диалог с сохранением контекста."""
    session_id = "conversation_001"
    
    messages = [
        "Как принять накладную?",
        "А если товар маркированный?",
        "Спасибо"
    ]
    
    for msg in messages:
        payload = {
            "message": msg,
            "session_id": session_id
        }
        
        response = requests.post(API_URL, json=payload, timeout=30)
        data = response.json()
        
        print(f"\nВопрос: {msg}")
        print(f"Ответ: {data['answer'][:200]}...")


# ═══════════════════════════════════════════════════════════════
# Пример 4: Интеграция в desktop-приложение (синхронно)
# ═══════════════════════════════════════════════════════════════

class FarmbazisSupportClient:
    """Клиент для интеграции в ПО Фармбазис."""
    
    def __init__(self, api_url=API_URL):
        self.api_url = api_url
        self.session_id = None
    
    def ask(self, question: str) -> dict:
        """
        Отправить вопрос боту.
        
        Returns:
            dict с полями:
                - type: "answer" | "clarification"
                - text: текст ответа
                - topics: list[dict] если нужно уточнение
                - confidence: float
                - youtube: list[str]
                - sources: list[str]
        """
        payload = {
            "message": question,
            "session_id": self.session_id
        }
        
        try:
            response = requests.post(
                self.api_url, 
                json=payload, 
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            
            # Сохраняем session_id для следующих запросов
            self.session_id = data.get("session_id")
            
            if data["response_type"] == "clarification":
                return {
                    "type": "clarification",
                    "text": data["answer"],
                    "topics": [
                        {
                            "number": i,
                            "title": t["title"],
                            "preview": t["snippet"]
                        }
                        for i, t in enumerate(data["suggested_topics"], 1)
                    ]
                }
            else:
                return {
                    "type": "answer",
                    "text": data["answer"],
                    "confidence": data["confidence"],
                    "confidence_level": data.get("confidence_level", ""),
                    "confidence_label": data.get("confidence_label", ""),
                    "youtube": data.get("youtube_links", []),
                    "sources": data.get("source_articles", []),
                    "needs_operator": data.get("needs_escalation", False)
                }
        
        except requests.exceptions.RequestException as e:
            return {
                "type": "error",
                "text": f"Ошибка соединения с сервером: {e}"
            }
        except Exception as e:
            return {
                "type": "error",
                "text": f"Неожиданная ошибка: {e}"
            }
    
    def reset(self):
        """Сбросить контекст диалога."""
        self.session_id = None


# ═══════════════════════════════════════════════════════════════
# Пример 5: Использование в UI
# ═══════════════════════════════════════════════════════════════

def ui_example():
    """Пример интеграции в форму техподдержки."""
    client = FarmbazisSupportClient()
    
    # Пользователь вводит вопрос
    user_question = "Ошибка при проведении накладной"
    
    result = client.ask(user_question)
    
    if result["type"] == "clarification":
        # Показываем кнопки с темами
        print("Выберите подходящую тему:")
        for topic in result["topics"]:
            print(f"  [{topic['number']}] {topic['title']}")
            # В UI: создать кнопку/радиобаттон
        
        # Пользователь выбрал тему 2
        selected = "2"
        result = client.ask(selected)
    
    if result["type"] == "answer":
        # Показываем ответ
        print(f"\nОтвет ({result['confidence']:.0%} уверенности, {result['confidence_level']} — {result['confidence_label']}):") 
        print(result["text"])
        
        if result["youtube"]:
            print("\nВидео-инструкции:")
            for link in result["youtube"]:
                print(f"  {link}")
                # В UI: встроить видео-плеер или кнопку "Смотреть"
        
        if result["needs_operator"]:
            print("\n⚠️ Рекомендуется обратиться к оператору")
            # В UI: кнопка "Связаться с оператором"


# ═══════════════════════════════════════════════════════════════
# Пример 6: C# / .NET интеграция (псевдокод)
# ═══════════════════════════════════════════════════════════════

"""
// C# пример для Фармбазис
using System.Net.Http;
using System.Text.Json;

public class FarmbazisAIClient
{
    private readonly HttpClient _http;
    private string _sessionId;
    
    public FarmbazisAIClient()
    {
        _http = new HttpClient
        {
            BaseAddress = new Uri("http://41.216.182.31:8000"),
            Timeout = TimeSpan.FromSeconds(30)
        };
    }
    
    public async Task<BotResponse> AskAsync(string question)
    {
        var request = new
        {
            message = question,
            session_id = _sessionId
        };
        
        var response = await _http.PostAsJsonAsync("/api/chat", request);
        response.EnsureSuccessStatusCode();
        
        var data = await response.Content.ReadFromJsonAsync<ChatApiResponse>();
        _sessionId = data.SessionId;
        
        return new BotResponse
        {
            Type = data.ResponseType,
            Text = data.Answer,
            Topics = data.SuggestedTopics,
            Confidence = data.Confidence
        };
    }
}

// Использование в форме
private async void btnAsk_Click(object sender, EventArgs e)
{
    var client = new FarmbazisAIClient();
    var result = await client.AskAsync(txtQuestion.Text);
    
    if (result.Type == "clarification")
    {
        // Показать RadioButton список тем
        foreach (var topic in result.Topics)
        {
            radioPanel.Controls.Add(new RadioButton
            {
                Text = topic.Title,
                Tag = topic.Number
            });
        }
    }
    else
    {
        // Показать ответ в RichTextBox
        txtAnswer.Text = result.Text;
    }
}
"""

# ═══════════════════════════════════════════════════════════════
# Запуск примеров
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Пример 1: Простой вопрос ===")
    simple_question()
    
    print("\n\n=== Пример 2: Уточнение ===")
    clarification_flow()
    
    print("\n\n=== Пример 4: Использование класса ===")
    ui_example()
