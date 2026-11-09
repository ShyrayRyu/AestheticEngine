<script setup lang="ts">
import { cn } from '@/lib/utils'

interface Props {
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  class: '',
})
</script>

<template>
  <div
    :class="
      cn(
        'rounded-2xl border border-zinc-800/80 bg-zinc-900/70 backdrop-blur-md text-zinc-100 shadow-2xl shadow-black/60',
        props.class
      )
    "
  >
    <slot />
  </div>
</template>
