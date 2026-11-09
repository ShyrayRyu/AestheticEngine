<script setup lang="ts">
import VideoGeneratorStudio from '@/components/VideoGeneratorStudio.vue'
</script>

<template>
  <main class="min-h-screen bg-zinc-950 text-zinc-100 selection:bg-red-600 selection:text-white">
    <!-- Halo lumineux subtil style Dark Gym Aesthetic -->
    <div
      class="pointer-events-none fixed inset-x-0 top-0 -z-10 h-96 bg-gradient-to-b from-red-950/20 via-zinc-950/5 to-transparent"
    />
    <VideoGeneratorStudio />
  </main>
</template>
