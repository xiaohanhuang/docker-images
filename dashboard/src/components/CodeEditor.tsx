
import Editor from '@monaco-editor/react';
import { Play } from 'lucide-react';
import { useState } from 'react';
import { api } from '@/lib/api';

export default function CodeEditor({ deskId }: { deskId: string }) {
  const code = `import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Model()
print(model)
`;

  const [editorCode, setEditorCode] = useState(code);
  const [isRunning, setIsRunning] = useState(false);

  const handleRun = async () => {
    if (!deskId || isRunning) return;
    setIsRunning(true);
    try {
      // Use internal next.js api proxy matching rest of the app
      const response = await fetch(`/api/v1/desks/${deskId}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: editorCode })
      });
      const result = await response.json();
      console.log('Run result:', result);
      // Dispatch event for the Terminal to display execution results inherently
      window.dispatchEvent(new CustomEvent('IDE_RUN', { 
         detail: { 
            stdout: result.stdout || '', 
            stderr: result.stderr || '',
            returncode: result.returncode
         } 
      }));
    } catch (err) {
      console.error('Failed to run code:', err);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '8px 16px', borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', justifyContent: 'flex-end', background: '#0a0d14' }}>
        <button 
          onClick={handleRun}
          disabled={isRunning}
          className="btn btn-primary btn-sm" 
          style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '4px 10px', opacity: isRunning ? 0.7 : 1 }}
        >
          <Play style={{ width: 12, height: 12 }} /> {isRunning ? 'Running...' : 'Run Code'}
        </button>
      </div>
      <div style={{ flex: 1, position: 'relative' }}>
        <Editor
          height="100%"
          defaultLanguage="python"
          value={editorCode}
          onChange={(val) => setEditorCode(val || '')}
        theme="vs-dark"
        options={{
          minimap: { enabled: false },
          fontSize: 14,
          fontFamily: 'Menlo, Monaco, "Courier New", monospace',
          padding: { top: 16 },
        }}
      />
      </div>
    </div>
  );
}
