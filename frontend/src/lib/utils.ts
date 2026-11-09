import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

/**
 * Utilitaire standard Shadcn-vue pour fusionner dynamiquement les classes Tailwind.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
