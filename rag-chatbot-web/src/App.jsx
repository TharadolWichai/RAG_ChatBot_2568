import { useState } from "react";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const askBot = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setAnswer("");
    setError("");

    try {
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        throw new Error(`Server responded with ${res.status}`);
      }

      const data = await res.json();

      if (data.answer) {
        setAnswer(data.answer);
      } else {
        setError("❌ ไม่มีคำตอบจากเซิร์ฟเวอร์");
      }
    } catch (err) {
      console.error(err);
      setError("❌ ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f8fafc",
        padding: 40,
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <div
        style={{
          maxWidth: 800,
          margin: "auto",
          background: "white",
          padding: 30,
          borderRadius: 12,
          boxShadow: "0 10px 25px rgba(0,0,0,0.08)",
        }}
      >
        <h2 style={{ marginBottom: 10 }}>🎓 RAG Chatbot – KKU CP</h2>
        <p style={{ color: "#64748b", marginBottom: 20 }}>
          Hybrid RAG (Rule-Based + LLM + Multi-Agent)
        </p>

        {/* Question */}
        <textarea
          rows={4}
          placeholder="พิมพ์คำถาม เช่น กลุ่มวิจัย AIDA คืออะไร?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          style={{
            width: "100%",
            padding: 12,
            borderRadius: 8,
            border: "1px solid #cbd5f5",
            marginBottom: 12,
            resize: "vertical",
            fontSize: 15,
          }}
        />

        {/* Button */}
        <button
          onClick={askBot}
          disabled={loading}
          style={{
            padding: "10px 24px",
            backgroundColor: loading ? "#a5b4fc" : "#4f46e5",
            color: "white",
            border: "none",
            borderRadius: 8,
            cursor: loading ? "not-allowed" : "pointer",
            fontSize: 15,
          }}
        >
          {loading ? "🤖 กำลังวิเคราะห์..." : "ถามบอท"}
        </button>

        {/* Answer */}
        {answer && (
          <div
            style={{
              marginTop: 24,
              background: "#f1f5f9",
              padding: 20,
              borderRadius: 10,
              whiteSpace: "pre-wrap", // ⭐ สำคัญ: แสดงบรรทัดใหม่
              lineHeight: 1.6,
            }}
          >
            <b>🤖 คำตอบ:</b>
            <div style={{ marginTop: 10 }}>{answer}</div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div
            style={{
              marginTop: 20,
              background: "#fee2e2",
              padding: 15,
              borderRadius: 8,
              color: "#b91c1c",
            }}
          >
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
