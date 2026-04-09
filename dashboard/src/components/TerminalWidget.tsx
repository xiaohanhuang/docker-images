
import { useEffect, useRef } from 'react';
import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';

export default function TerminalWidget({ deskId }: { deskId: string }) {
  const terminalRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  
  useEffect(() => {
    if (!terminalRef.current) return;
    
    const term = new Terminal({
      theme: {
        background: '#0a0d14',
        foreground: '#e2e8f0',
        cursor: '#7c3aed'
      },
      fontFamily: 'Menlo, Monaco, "Courier New", monospace',
      fontSize: 13,
      cursorBlink: true
    });
    
    const fitAddon = new FitAddon();
    term.loadAddon(fitAddon);
    
    term.open(terminalRef.current);
    fitAddon.fit();
    
    term.writeln('Welcome to Desk Terminal...');
    term.writeln(`Connecting to pod: ${deskId}...`);
    
    // Connect to WebSocket via the Next.js proxy rewrite
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    // Proxies through Next.js rewrite (/api/v1/desks -> backend:8000/api/v1/desks)
    const wsUrl = `${protocol}//${host}/api/v1/desks/${deskId}/logs`;
    
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      term.writeln('\r\nConnected to live kubectl logs \r\n');
    };

    ws.onmessage = (event) => {
      // The backend sends string lines, replace \n with \r\n for xterm
      term.write(event.data.replace(/\n/g, '\r\n'));
    };

    ws.onerror = (error) => {
      term.writeln(`\r\n\u001b[31mTerminal WebSocket error: ${error}\u001b[0m\r\n`);
    };

    ws.onclose = () => {
      term.writeln('\r\n\u001b[33mTerminal connection closed.\u001b[0m\r\n');
    };
    
    const handleResize = () => {
      fitAddon.fit();
    };
    
    const handleRunResult = (e: any) => {
      const { stdout, stderr, returncode } = e.detail;
      term.writeln('\r\n\u001b[36m--- Executing Python Script ---\u001b[0m\r\n');
      if (stdout) {
         term.write(stdout.replace(/\n/g, '\r\n'));
      }
      if (stderr) {
         term.write('\u001b[31m' + stderr.replace(/\n/g, '\r\n') + '\u001b[0m');
      }
      term.writeln(`\r\n\u001b[33mProcess exited with code ${returncode}\u001b[0m\r\n`);
    };

    window.addEventListener('resize', handleResize);
    window.addEventListener('IDE_RUN', handleRunResult);

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('IDE_RUN', handleRunResult);
      term.dispose();
      if (wsRef.current) wsRef.current.close();
    };
  }, [deskId]);

  return <div ref={terminalRef} style={{ width: '100%', height: '100%' }} />;
}
