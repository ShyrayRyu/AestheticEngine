<script setup lang="ts">
import { computed } from 'vue'
import { cn } from '@/lib/utils'

interface Props {
  variant?: 'default' | 'secondary' | 'outline' | 'ghost' | 'destructive'
  size?: 'default' | 'sm' | 'lg' | 'icon'
  disabled?: boolean
  type?: 'button' | 'submit' | 'reset'
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'default',
  size: 'default',
  disabled: false,
  type: 'button',
  class: '',
})

const buttonClass = computed(() => {
  const base =
    'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 disabled:pointer-events-none disabled:opacity-50'

  const variants: Record<NonNullable<Props['variant']>, string> = {
    default:
      'bg-gradient-to-r from-red-600 to-rose-700 text-white shadow-lg shadow-red-950/50 hover:from-red-500 hover:to-rose-600',
    secondary: 'bg-zinc-800 text-zinc-100 hover:bg-zinc-700 border border-zinc-700',
    outline:
      'border border-zinc-700 bg-zinc-900/60 text-zinc-200 hover:bg-zinc-800 hover:text-white',
    ghost: 'text-zinc-400 hover:bg-zinc-800/70 hover:text-zinc-100',
    destructive: 'bg-red-900/80 text-red-100 hover:bg-red-800',
  }

  const sizes: Record<NonNullable<Props['size']>, string> = {
    default: 'h-10 px-4 py-2',
    sm: 'h-8 rounded-md px-3 text-xs',
    lg: 'h-12 rounded-xl px-6 text-base',
    icon: 'h-9 w-9',
  }

  return cn(base, variants[props.variant], sizes[props.size], props.class)
})
</script>

<template>
  <button :type="props.type" :disabled="props.disabled" :class="buttonClass">
    <slot />
  </button>
</template>
