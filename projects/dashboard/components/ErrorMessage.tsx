interface ErrorMessageProps {
  message: string;
}

export function ErrorMessage({ message }: ErrorMessageProps) {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="text-center">
        <p className="text-red-600 font-medium">Error loading data</p>
        <p className="text-gray-500 text-sm mt-2">{message}</p>
      </div>
    </div>
  );
}
