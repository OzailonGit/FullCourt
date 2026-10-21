import { useRef, useState, type FormEvent } from 'react'

export type Availability = 'loading' | 'ready' | 'unavailable'

interface QuestionFormProps {
  question: string
  examples: string[]
  availability: Availability
  searching: boolean
  onQuestionChange: (question: string) => void
  onSearch: (question: string) => void
}

const EMPTY_QUESTION_ERROR = 'Type a question first, or pick one of the examples.'

const BUTTON_LABELS: Record<Availability, string> = {
  loading: 'Loading…',
  ready: 'Search',
  unavailable: 'Unavailable',
}

export function QuestionForm({ question, examples, availability, searching, onQuestionChange, onSearch }: QuestionFormProps) {
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const canSearch = availability === 'ready' && !searching

  const submit = (event: FormEvent) => {
    event.preventDefault()
    if (!canSearch) return
    if (!question.trim()) {
      setError(EMPTY_QUESTION_ERROR)
      inputRef.current?.focus()
      return
    }
    setError(null)
    onSearch(question)
  }

  return (
    <section className="question" aria-label="Ask a question">
      <form onSubmit={submit} className="question-form" noValidate>
        <label htmlFor="question" className="question-label">Question</label>
        <div className="question-row">
          <input
            ref={inputRef}
            id="question"
            type="text"
            value={question}
            maxLength={500}
            autoComplete="off"
            placeholder="Which guards attack the rim and get to the line?"
            aria-invalid={error ? true : undefined}
            aria-describedby={error ? 'question-error' : undefined}
            onChange={(event) => {
              if (error) setError(null)
              onQuestionChange(event.target.value)
            }}
          />
          {/* aria-disabled rather than disabled: a disabled button drops keyboard focus to the top of
              the page in the middle of a search. The submit handler ignores presses while busy. */}
          <button type="submit" className="button-primary" aria-disabled={!canSearch}>
            {searching ? 'Searching…' : BUTTON_LABELS[availability]}
          </button>
        </div>
        {error && <p id="question-error" className="field-error" role="alert">{error}</p>}
      </form>

      {examples.length > 0 && (
        <details className="disclosure examples">
          <summary>Example questions</summary>
          <ul className="example-list">
            {examples.map((example) => (
              <li key={example}>
                <button
                  type="button"
                  className="example"
                  aria-disabled={!canSearch}
                  onClick={() => {
                    if (!canSearch) return
                    setError(null)
                    onQuestionChange(example)
                    onSearch(example)
                  }}
                >
                  {example}
                </button>
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  )
}
