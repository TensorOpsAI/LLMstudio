'use client';

import * as React from 'react';
import { Toaster as SonnerToaster } from 'sonner';
import { useTheme } from 'next-themes';

export function Toaster() {
  const { theme } = useTheme();
  return (
    <SonnerToaster
      richColors
      theme={theme as 'light' | 'dark' | 'system' | undefined}
    />
  );
}
