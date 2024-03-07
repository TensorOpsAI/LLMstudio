'use client';
import React from 'react';
import { Player } from '@lottiefiles/react-lottie-player';
import { Toggle } from '@/components/Theme';

import Link from 'next/link';
import { Shapes, GitCompare, BarChart3, BookAIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';

const links = [
  {
    name: 'Playground',
    route: '/playground',
    icon: Shapes,
  },
  // {
  //   name: 'Compare',
  //   route: '/compare',
  //   icon: GitCompare,
  // },
  {
    name: 'Dashboard',
    route: '/dashboard',
    icon: BarChart3,
  },
  {
    name: 'Docs',
    route: 'https://docs.llmstudio.ai',
    icon: BookAIcon,
  },
];

export default function Header() {
  return (
    <header className='relative flex h-20 p-4'>
      <div className='inline-flex w-[200px] items-center'>
        <Link href='/'>
          <Player
            hover
            loop
            src='logo.json'
            className='invert transition dark:invert-0'
            style={{ height: '100%', width: '100%' }}
          />
        </Link>
      </div>
      <div className='absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 items-center gap-4'>
        {links.map((link) => (
          <Button asChild key={link.route} variant='ghost'>
            <Link href={link.route}>
              {React.createElement(link.icon, { className: 'mr-2 h-5 w-5' })}
              {link.name}
            </Link>
          </Button>
        ))}
        <Toggle />
      </div>
    </header>
  );
}
