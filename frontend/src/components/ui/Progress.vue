<script setup lang="ts">
import { computed } from 'vue'
import { cn } from '@/lib/utils'

interface Props {
  modelValue?: number
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  modelValue: 0,
  class: '',
})

const clampedPercent = computed(() => Math.max(0, Math.min(100, props.modelValue)))
</script>

<template>
  <div
    role="progressbar"
    :aria-valuenow="clampedPercent"
    aria-valuemin="0"
    aria-valuemax="100"
    :class="
      cn(
        'relative h-3.5 w-full overflow-hidden rounded-full bg-zinc-950 border border-zinc-800',
        props.class
      )
    "
  >
    <div
      class="h-full bg-gradient-to-r from-red-600 via-rose-500 to-amber-500 transition-all duration-500 ease-out"
      :style="{ width: `${clampedPercent}%` }"
    />
  </div>
</template>
