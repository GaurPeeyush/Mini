import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'

const API_BASE = (import.meta as any).env.VITE_API_BASE || 'http://localhost:8000'

type HistoryItem = { ts:number, question:string, answer:string, source:string }

type AskResponse = {
  answer: string
  source: 'kb' | 'llm'
  matchedQuestion?: string | null
  score?: number | null
  trace?: string[] | null
}

function Badge({label}:{label:string}){
  return <span className="chip">{label}</span>
}

function App(){
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<AskResponse | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function fetchHistory(){
    try{
      const res = await fetch(`${API_BASE}/history?limit=10`)
      const data = await res.json()
      setHistory(data.items || [])
    }catch{ /* ignore */ }
  }

  useEffect(()=>{ fetchHistory() }, [])

  async function onAsk(e:React.FormEvent){
    e.preventDefault()
    if(!question.trim()) return
    setLoading(true); setError(null); setAnswer(null)
    try{
      const res = await fetch(`${API_BASE}/ask`,{
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ question })
      })
      if(!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: AskResponse = await res.json()
      setAnswer(data); setQuestion('')
      await fetchHistory()
    }catch(err:any){ setError(err.message||'Failed') }
    finally{ setLoading(false) }
  }

  async function onClearHistory(){
    try{
      await fetch(`${API_BASE}/history/clear`, { method:'POST' })
      setHistory([])
    }catch{ /* ignore */ }
  }

  const scoreText = useMemo(()=>{
    if(!answer || answer.score==null) return '—'
    return answer.score.toFixed(4)
  },[answer])

  return (
    <div className="container">
      <div className="card">
        <div className="header">
          <div className="brand">
            <div style={{width:10,height:10,borderRadius:999,background:'#22c55e'}} />
            <strong>Mini AI Chatbot</strong>
            <span className="brand-badge">Pinecone • SentenceTransformers • Claude</span>
          </div>
          <div className="chips">
            <Badge label="KB" />
            <Badge label="Cosine 384-d" />
            <Badge label="Serverless" />
          </div>
        </div>

        <div className="main">
          <form onSubmit={onAsk} className="row">
            <input className="input" value={question} onChange={e=>setQuestion(e.target.value)} placeholder="Ask a professional question..." />
            <button className="btn" disabled={loading || !question.trim()}>{loading ? 'Thinking...' : 'Ask'}</button>
            <button type="button" className="btn outline" onClick={onClearHistory}>Clear history</button>
          </form>

          {error && <div style={{color:'#fca5a5', marginTop:8}}>{error}</div>}

          {answer && (
            <div className="answer">
              <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
                <strong>Answer</strong>
                <div className="chips">
                  <Badge label={`Source: ${answer.source}`} />
                  <Badge label={`Score: ${scoreText}`} />
                </div>
              </div>
              <div style={{marginTop:8, whiteSpace:'pre-wrap'}}>{answer.answer}</div>

              <details>
                <summary>How I answered</summary>
                <div className="trace">
                  {answer.matchedQuestion && (
                    <div style={{marginBottom:8}}>
                      Matched question: <em>{answer.matchedQuestion}</em>
                    </div>
                  )}
                  <ul>
                    {(answer.trace||[]).map((t,i)=> <li key={i}>{t}</li>)}
                  </ul>
                </div>
              </details>
            </div>
          )}

          <div style={{marginTop:24}}>
            <strong>Chat History (Last 10)</strong>
            <ul>
              {history.map((h,i)=> (
                <li key={i}>
                  <span style={{color:'#94a3b8'}}>{new Date(h.ts*1000).toLocaleString()}:</span> <em>{h.question}</em>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="footer">Built with Pinecone Vector DB and Claude fallback • Demo</div>
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<App />) 