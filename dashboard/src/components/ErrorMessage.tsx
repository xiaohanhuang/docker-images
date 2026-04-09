interface ErrorMessageProps {
  message: string;
}

export function ErrorMessage({ message }: ErrorMessageProps) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '16rem' }}>
      <div className="text-center">
        <p className="text-red-600 font-medium">Error loading data</p>
        <p className="text-gray-500 text-sm mt-2">{message}</p>
      </div>
    </div>
  );
}
