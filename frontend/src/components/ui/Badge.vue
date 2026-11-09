<script setup lang="ts">
import { computed } from 'vue'
import { cn } from '@/lib/utils'

interface Props {
  variant?: 'default' | 'secondary' | 'outline' | 'success'
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'default',
  class: '',
})

const badgeClass = computed(() => {
  const variants: Record<NonNullable<Props['variant']>, string> = {
    default: 'bg-red-950/90 text-red-300 border-red-800/70',
    secondary: 'bg-zinc-800 text-zinc-300 border-zinc-700',
    outline: 'bg-transparent text-zinc-400 border-zinc-700',
    success: 'bg-emerald-950/80 text-emerald-300 border-emerald-800/70',
  }
  return cn(
    'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors',
    variants[props.variant],
    props.class
  )
})
</script>

<template>
  <span :class="badgeClass">
    <slot />
  </span>
</template>
