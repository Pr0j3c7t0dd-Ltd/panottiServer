import React from 'react';
import { XCircleIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';

interface ErrorDisplayProps {
  title: string;
  message: string;
  severity?: 'error' | 'warning';
  retry?: () => void;
}

const ErrorDisplay: React.FC<ErrorDisplayProps> = ({
  title,
  message,
  severity = 'error',
  retry,
}) => {
  const Icon = severity === 'error' ? XCircleIcon : ExclamationTriangleIcon;
  const bgColor = severity === 'error' ? 'bg-red-50' : 'bg-yellow-50';
  const textColor = severity === 'error' ? 'text-red-700' : 'text-yellow-700';
  const buttonColor = severity === 'error' ? 'bg-red-600 hover:bg-red-500' : 'bg-yellow-600 hover:bg-yellow-500';

  return (
    <div className={`rounded-md ${bgColor} p-4`}>
      <div className="flex">
        <div className="shrink-0">
          <div className="size-5">
            <Icon className={`h-5 w-5 ${textColor}`} aria-hidden="true" />
          </div>
        </div>
        <div className="ml-3">
          <h3 className={`text-sm font-medium ${textColor}`}>{title}</h3>
          <div className={`mt-2 text-sm ${textColor}`}>
            <p>{message}</p>
          </div>
          {retry && (
            <div className="mt-4">
              <div className="-mx-2 -my-1.5 flex">
                <button
                  type="button"
                  onClick={retry}
                  className={`rounded-md px-2 py-1.5 text-sm font-medium text-white ${buttonColor} focus:outline-none focus:ring-2 focus:ring-offset-2`}
                >
                  Retry
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ErrorDisplay;
