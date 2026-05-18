import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import SourceCitation from './SourceCitation';
import type { ChatMessage as ChatMessageType } from '../../api/types';

interface Props {
  message: ChatMessageType;
  isStreaming?: boolean;
}

export default function ChatMessage({ message, isStreaming }: Props) {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="chat-message-user">
        <div className="chat-bubble-user">{message.content}</div>
      </div>
    );
  }

  return (
    <div className="chat-message-assistant">
      <div>
        <div className={`chat-bubble-assistant ${isStreaming ? 'streaming-cursor' : ''}`}>
          <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
        </div>
        {message.sources && message.sources.length > 0 && (
          <SourceCitation sources={message.sources} />
        )}
      </div>
    </div>
  );
}
