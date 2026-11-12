<script setup lang="ts">
/**
 * Composant principal Vue 3 - VideoGeneratorStudio.vue
 * ----------------------------------------------------
 * Studio "Aesthetic Posing & Bodybuilding" permettant :
 *  1. Le glisser-déposer (Drag & Drop) d'un ou plusieurs rushs de posing (.mp4) et d'un son (.mp3/.wav)
 *  2. Le choix du format (9:16 Vertical ou 16:9 Paysage) avec rognage centré strict SANS ÉTIREMENT
 *  3. Le choix du mode de coupe :
 *       - "Posing Cut 1.0x (Style exemple.mp4)" : coupes franches sans transition sur les beats,
 *         détection multi-poses variées (Intro + Most Muscular, Side Chest, Double Biceps...)
 *       - "Speed Ramping" : accélération d'amorce + ralenti sur le drop
 *  4. Le suivi en temps réel de la génération, la prévisualisation vidéo directe et l'export .mp4 / .xml
 */

import { computed, onUnmounted, reactive, ref } from 'vue'
import {
  Activity,
  CheckCircle2,
  Cpu,
  Download,
  FileCode2,
  Film,
  Flame,
  Loader2,
  Music,
  Sparkles,
  Trash2,
  UploadCloud,
  Video,
  Wand2,
  AlertTriangle,
  Maximize2,
  Scissors,
  Eye,
  VolumeX,
} from 'lucide-vue-next'

import Button from '@/components/ui/Button.vue'
import Card from '@/components/ui/Card.vue'
import Progress from '@/components/ui/Progress.vue'
import Badge from '@/components/ui/Badge.vue'
import {
  API_BASE_URL,
  WS_BASE_URL,
  fetchProjectStatus,
  startProjectProcessing,
  uploadProjectMedia,
} from '@/services/api'
import type {
  PipelineStage,
  ProcessProjectRequest,
  ProjectStatusResponse,
} from '@/types/project'

// ---------------------------------------------------------------------------
// ÉTAT RÉACTIF : FICHIERS DRAG-AND-DROP
// ---------------------------------------------------------------------------
const videoFiles = ref<File[]>([])
const audioFile = ref<File | null>(null)
const inspirationFile = ref<File | null>(null)

const isDraggingVideos = ref(false)
const isDraggingAudio = ref(false)
const isDraggingInspiration = ref(false)

const videoInputRef = ref<HTMLInputElement | null>(null)
const audioInputRef = ref<HTMLInputElement | null>(null)
const inspirationInputRef = ref<HTMLInputElement | null>(null)

// ---------------------------------------------------------------------------
// ÉTAT RÉACTIF : PARAMÈTRES DU MOTEUR POSING & BEAT-CUT
// ---------------------------------------------------------------------------
const options = reactive<ProcessProjectRequest>({
  aspect_ratio: '9:16',
  auto_orient_and_frame: true,
  editing_mode: 'pose_cut',
  target_duration: 21.5,
  include_intro_shot: true,
  min_effort_score: 7.0,
  ollama_model: 'llava:latest',
  fast_speed_factor: 1.35,
  slow_speed_factor: 0.75,
  lut_preset: 'natural_aesthetic',
  drop_sensitivity: 0.65,
  use_inspiration_audio: true,
  mute_output: false,
})

// ---------------------------------------------------------------------------
// ÉTAT RÉACTIF : PROGRESSION DU PIPELINE & CONNEXION TEMPS-RÉEL
// ---------------------------------------------------------------------------
const isUploading = ref(false)
const isProcessing = ref(false)
const uiError = ref<string | null>(null)
const projectStatus = ref<ProjectStatusResponse | null>(null)

let activeSocket: WebSocket | null = null
let pollingTimer: ReturnType<typeof setInterval> | null = null

const PIPELINE_STEPS: Array<{
  key: PipelineStage
  label: string
  sublabel: string
}> = [
  {
    key: 'analyzing_audio',
    label: '1. Analyse Audio (Librosa)',
    sublabel: 'Détection BPM & Points de coupe Beat Drops',
  },
  {
    key: 'analyzing_vision',
    label: '2. Détection des Poses (IA / CV)',
    sublabel: 'Isolation Intro + Poses verrouillées variées',
  },
  {
    key: 'speed_ramping',
    label: '3. Montage Beat-Cut (Sans Transition)',
    sublabel: 'Rognage propre 9:16 ou 16:9 sans étirement',
  },
  {
    key: 'color_grading',
    label: '4. Rendu & Étalonnage FFmpeg',
    sublabel: 'Clarté musculaire & respect de la lumière',
  },
  {
    key: 'exporting_edl',
    label: '5. Export MP4 & XML (EDL)',
    sublabel: 'Master H.264 + Timeline Premiere / Resolve',
  },
]

// ---------------------------------------------------------------------------
// PROPRIÉTÉS CALCULÉES (COMPUTED)
// ---------------------------------------------------------------------------
const canLaunchPipeline = computed(() => {
  return (
    videoFiles.value.length > 0 &&
    !isUploading.value &&
    !isProcessing.value
  )
})

const progressPercent = computed(() => projectStatus.value?.progress_percent ?? 0)

const finalVideoFullUrl = computed(() => {
  if (!projectStatus.value?.output_video_url) return null
  return `${API_BASE_URL}${projectStatus.value.output_video_url}?t=${Date.now()}`
})

const finalXmlFullUrl = computed(() => {
  if (!projectStatus.value?.output_xml_url) return null
  return `${API_BASE_URL}${projectStatus.value.output_xml_url}`
})

// ---------------------------------------------------------------------------
// GESTIONNAIRES DRAG-AND-DROP (VIDÉOS & AUDIO)
// ---------------------------------------------------------------------------
function handleVideoDrop(event: DragEvent) {
  isDraggingVideos.value = false
  const droppedFiles = event.dataTransfer?.files
  if (!droppedFiles) return
  addVideoFiles(Array.from(droppedFiles))
}

function handleVideoSelect(event: Event) {
  const target = event.target as HTMLInputElement
  if (!target.files) return
  addVideoFiles(Array.from(target.files))
  target.value = ''
}

function addVideoFiles(files: File[]) {
  uiError.value = null
  const validVideos = files.filter(
    (f) =>
      f.type.startsWith('video/') ||
      /\.(mp4|mov|mkv|webm)$/i.test(f.name)
  )
  if (validVideos.length === 0) {
    uiError.value = 'Veuillez déposer uniquement des fichiers vidéo (.mp4, .mov).'
    return
  }
  videoFiles.value.push(...validVideos)
}

function removeVideoFile(index: number) {
  videoFiles.value.splice(index, 1)
}

function handleAudioDrop(event: DragEvent) {
  isDraggingAudio.value = false
  const droppedFiles = event.dataTransfer?.files
  if (!droppedFiles || droppedFiles.length === 0) return
  setAudioTrack(droppedFiles[0])
}

function handleAudioSelect(event: Event) {
  const target = event.target as HTMLInputElement
  if (!target.files || target.files.length === 0) return
  setAudioTrack(target.files[0])
  target.value = ''
}

function setAudioTrack(file: File) {
  uiError.value = null
  if (
    !file.type.startsWith('audio/') &&
    !/\.(mp3|wav|m4a|aac|flac|ogg)$/i.test(file.name)
  ) {
    uiError.value = 'Veuillez sélectionner un fichier audio valide (.mp3, .wav, .m4a).'
    return
  }
  audioFile.value = file
  options.mute_output = false
}

function handleInspirationDrop(event: DragEvent) {
  isDraggingInspiration.value = false
  const droppedFiles = event.dataTransfer?.files
  if (!droppedFiles || droppedFiles.length === 0) return
  setInspirationVideo(droppedFiles[0])
}

function handleInspirationSelect(event: Event) {
  const target = event.target as HTMLInputElement
  if (!target.files || target.files.length === 0) return
  setInspirationVideo(target.files[0])
  target.value = ''
}

function setInspirationVideo(file: File) {
  uiError.value = null
  if (
    !file.type.startsWith('video/') &&
    !/\.(mp4|mov|mkv|webm)$/i.test(file.name)
  ) {
    uiError.value = "Veuillez sélectionner une vidéo d'inspiration valide (.mp4, .mov)."
    return
  }
  inspirationFile.value = file
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} Ko`
  return `${(bytes / (1024 * 1024)).toFixed(1)} Mo`
}

// ---------------------------------------------------------------------------
// LANCEMENT DE L'UPLOAD ET DU PIPELINE DE MONTAGE
// ---------------------------------------------------------------------------
async function launchAutomatedEdit() {
  if (!canLaunchPipeline.value) return

  uiError.value = null
  isUploading.value = true

  try {
    const uploadRes = await uploadProjectMedia(
      videoFiles.value,
      audioFile.value,
      inspirationFile.value
    )

    isUploading.value = false
    isProcessing.value = true

    const initialStatus = await startProjectProcessing(
      uploadRes.project_id,
      options
    )
    projectStatus.value = initialStatus

    subscribeToProjectProgress(uploadRes.project_id)
  } catch (err: unknown) {
    isUploading.value = false
    isProcessing.value = false
    uiError.value =
      err instanceof Error
        ? err.message
        : 'Une erreur inattendue est survenue.'
  }
}

function subscribeToProjectProgress(projectId: string) {
  cleanupConnections()

  try {
    activeSocket = new WebSocket(`${WS_BASE_URL}/ws/projects/${projectId}`)

    activeSocket.onmessage = (event) => {
      const data: ProjectStatusResponse = JSON.parse(event.data)
      projectStatus.value = data
      if (data.stage === 'completed' || data.stage === 'failed') {
        isProcessing.value = false
        cleanupConnections()
      }
    }

    activeSocket.onerror = () => {
      startFallbackPolling(projectId)
    }
  } catch {
    startFallbackPolling(projectId)
  }
}

function startFallbackPolling(projectId: string) {
  if (pollingTimer) return
  pollingTimer = setInterval(async () => {
    try {
      const data = await fetchProjectStatus(projectId)
      projectStatus.value = data
      if (data.stage === 'completed' || data.stage === 'failed') {
        isProcessing.value = false
        cleanupConnections()
      }
    } catch {
      // Ignorer les erreurs réseau transitoires
    }
  }, 1000)
}

function cleanupConnections() {
  if (activeSocket) {
    activeSocket.close()
    activeSocket = null
  }
  if (pollingTimer) {
    clearInterval(pollingTimer)
    pollingTimer = null
  }
}

function getStepState(
  stepKey: PipelineStage
): 'done' | 'active' | 'pending' {
  const current = projectStatus.value?.stage
  if (!current) return 'pending'
  if (current === 'completed') return 'done'

  const order: PipelineStage[] = [
    'analyzing_audio',
    'analyzing_vision',
    'speed_ramping',
    'color_grading',
    'exporting_edl',
  ]
  const currentIdx = order.indexOf(current)
  const stepIdx = order.indexOf(stepKey)

  if (currentIdx === -1) return 'pending'
  if (stepIdx < currentIdx) return 'done'
  if (stepIdx === currentIdx) return 'active'
  return 'pending'
}

onUnmounted(() => {
  cleanupConnections()
})
</script>

<template>
  <div class="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
    <!-- ===================================================================== -->
    <!-- EN-TÊTE DU STUDIO POSING & AESTHETIC                                   -->
    <!-- ===================================================================== -->
    <header
      class="mb-8 flex flex-col justify-between gap-4 border-b border-zinc-800/80 pb-6 md:flex-row md:items-center"
    >
      <div class="flex items-center gap-3.5">
        <div
          class="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-red-600 to-rose-900 shadow-lg shadow-red-950/60"
        >
          <Flame class="h-6 w-6 text-white" />
        </div>
        <div>
          <div class="flex items-center gap-2.5">
            <h1 class="text-2xl font-extrabold tracking-tight text-white sm:text-3xl">
              AESTHETIC<span class="text-red-500">ENGINE</span>
            </h1>
            <Badge variant="default">POSING & MUSCLE CUT STUDIO</Badge>
          </div>
          <p class="mt-0.5 text-sm text-zinc-400">
            Montage TikTok Posing automatisé • Coupes franches sur les Beats (sans transition) •
            Rognage propre {{ options.aspect_ratio }} sans étirement
          </p>
        </div>
      </div>

      <div class="flex flex-wrap items-center gap-2">
        <Badge variant="secondary">
          <Maximize2 class="h-3.5 w-3.5 text-red-400" />
          Format : {{ options.aspect_ratio }} ({{
            options.aspect_ratio === '9:16' ? '1080×1920' : '1920×1080'
          }})
        </Badge>
        <Badge variant="secondary">
          <Scissors class="h-3.5 w-3.5 text-red-400" />
          Coupes : Sans transition (Hard Cuts)
        </Badge>
        <Badge variant="secondary">
          <Cpu class="h-3.5 w-3.5 text-red-400" />
          Vision : Posing Lock + {{ options.ollama_model }}
        </Badge>
      </div>
    </header>

    <!-- Bannière d'erreur éventuelle -->
    <div
      v-if="uiError || projectStatus?.error_message"
      class="mb-6 flex items-center gap-3 rounded-xl border border-red-800/80 bg-red-950/60 px-4 py-3 text-sm text-red-200"
    >
      <AlertTriangle class="h-5 w-5 shrink-0 text-red-400" />
      <span>{{ uiError || projectStatus?.error_message }}</span>
    </div>

    <!-- ===================================================================== -->
    <!-- GRILLE PRINCIPALE : DRAG & DROP (GAUCHE) + RÉGLAGES POSING (DROITE)    -->
    <!-- ===================================================================== -->
    <div class="grid grid-cols-1 gap-6 lg:grid-cols-12">
      <!-- COLONNE GAUCHE : UPLOAD DRAG-AND-DROP (7 colonnes) -->
      <div class="space-y-6 lg:col-span-7">
        <!-- 1. Zone de dépôt des Rushs Vidéo (.mp4) -->
        <Card class="p-6">
          <div class="mb-4 flex items-center justify-between">
            <div class="flex items-center gap-2.5">
              <Video class="h-5 w-5 text-red-500" />
              <h2 class="text-lg font-bold text-white">
                1. Rush(s) Vidéo de Posing / Musculation (.mp4)
              </h2>
            </div>
            <Badge variant="outline">
              {{ videoFiles.length }} fichier(s) sélectionné(s)
            </Badge>
          </div>

          <div
            class="group relative flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 text-center transition-all"
            :class="
              isDraggingVideos
                ? 'border-red-500 bg-red-950/20'
                : 'border-zinc-800 bg-zinc-950/50 hover:border-zinc-700 hover:bg-zinc-900/50'
            "
            @dragover.prevent="isDraggingVideos = true"
            @dragleave.prevent="isDraggingVideos = false"
            @drop.prevent="handleVideoDrop"
            @click="videoInputRef?.click()"
          >
            <input
              ref="videoInputRef"
              type="file"
              accept="video/mp4,video/quicktime,.mp4,.mov"
              multiple
              class="hidden"
              @change="handleVideoSelect"
            />

            <div
              class="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-zinc-900 border border-zinc-800 group-hover:border-red-500/50 transition-colors"
            >
              <UploadCloud class="h-6 w-6 text-red-500" />
            </div>
            <p class="text-sm font-medium text-zinc-200">
              Glissez-déposez un rush long (avec plusieurs poses) ou plusieurs rushs courts
            </p>
            <p class="mt-1 text-xs text-zinc-500">
              Détection automatique de chaque pose verrouillée (Most Muscular, Side Chest, Double Biceps...)
            </p>
          </div>

          <!-- Liste des rushs vidéo ajoutés -->
          <ul
            v-if="videoFiles.length > 0"
            class="mt-4 max-h-48 space-y-2 overflow-y-auto pr-1"
          >
            <li
              v-for="(file, idx) in videoFiles"
              :key="`${file.name}-${idx}`"
              class="flex items-center justify-between rounded-lg border border-zinc-800/90 bg-zinc-950/80 px-3.5 py-2 text-sm"
            >
              <div class="flex items-center gap-2.5 truncate">
                <Film class="h-4 w-4 shrink-0 text-red-400" />
                <span class="truncate font-medium text-zinc-200">
                  {{ file.name }}
                </span>
                <span class="text-xs text-zinc-500">
                  ({{ formatFileSize(file.size) }})
                </span>
              </div>
              <button
                type="button"
                class="text-zinc-500 hover:text-red-400 transition-colors"
                title="Retirer ce rush"
                @click.stop="removeVideoFile(idx)"
              >
                <Trash2 class="h-4 w-4" />
              </button>
            </li>
          </ul>
        </Card>

        <!-- 2. Zone de dépôt de la Piste Audio (Optionnelle) -->
        <Card class="p-6">
          <div class="mb-4 flex items-center justify-between">
            <div class="flex items-center gap-2.5">
              <Music class="h-5 w-5 text-red-500" />
              <h2 class="text-lg font-bold text-white">
                2. Bande-Son TikTok (.mp3, .wav) — Optionnel
              </h2>
            </div>
            <Badge v-if="audioFile" variant="success">Piste prête</Badge>
            <Badge v-else variant="secondary">
              Sans Musique (Son vidéo à 0)
            </Badge>
          </div>

          <div
            class="group relative flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition-all"
            :class="
              isDraggingAudio
                ? 'border-red-500 bg-red-950/20'
                : 'border-zinc-800 bg-zinc-950/50 hover:border-zinc-700 hover:bg-zinc-900/50'
            "
            @dragover.prevent="isDraggingAudio = true"
            @dragleave.prevent="isDraggingAudio = false"
            @drop.prevent="handleAudioDrop"
            @click="audioInputRef?.click()"
          >
            <input
              ref="audioInputRef"
              type="file"
              accept="audio/*,.mp3,.wav,.m4a"
              class="hidden"
              @change="handleAudioSelect"
            />

            <template v-if="!audioFile">
              <Music class="mb-2 h-6 w-6 text-red-500" />
              <p class="text-sm font-medium text-zinc-200">
                Déposez un son TikTok (.mp3, .wav) ou laissez vide pour monter sans musique
              </p>
              <p class="mt-1 text-xs text-zinc-500">
                Sans musique : les poses s'enchaînent automatiquement et le son de la vidéo est forcé à 0 (muet)
              </p>
            </template>

            <div
              v-else
              class="flex w-full items-center justify-between rounded-lg bg-zinc-900/90 px-4 py-2.5"
              @click.stop
            >
              <div class="flex items-center gap-3 truncate">
                <Music class="h-5 w-5 shrink-0 text-red-400" />
                <div class="text-left truncate">
                  <p class="truncate text-sm font-semibold text-white">
                    {{ audioFile.name }}
                  </p>
                  <p class="text-xs text-zinc-400">
                    {{ formatFileSize(audioFile.size) }}
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                @click="audioFile = null"
              >
                Retirer (Monter sans musique)
              </Button>
            </div>
          </div>

          <!-- Option explicite pour forcer le son à 0 (même avec une vidéo d'inspiration) -->
          <label
            v-if="!audioFile"
            class="mt-3 flex items-center justify-between rounded-xl border border-zinc-800/80 bg-zinc-950/60 px-3.5 py-2 cursor-pointer"
          >
            <div class="flex items-center gap-2">
              <VolumeX class="h-4 w-4 text-zinc-400" />
              <span class="text-xs text-zinc-300">
                Forcer le montage sans musique (volume de la vidéo finale à 0)
              </span>
            </div>
            <input
              v-model="options.mute_output"
              type="checkbox"
              class="h-4 w-4 accent-red-600 rounded"
            />
          </label>
        </Card>

        <!-- 3. Zone de dépôt d'une Vidéo d'Inspiration / Référence (Optionnelle) -->
        <Card class="p-6">
          <div class="mb-4 flex items-center justify-between">
            <div class="flex items-center gap-2.5">
              <Eye class="h-5 w-5 text-red-500" />
              <h2 class="text-lg font-bold text-white">
                3. Vidéo d'Inspiration / Référence (.mp4) — Optionnel
              </h2>
            </div>
            <Badge v-if="inspirationFile" variant="success">
              Modèle d'inspiration actif
            </Badge>
            <Badge v-else variant="outline">
              Facultatif
            </Badge>
          </div>

          <div
            class="group relative flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition-all"
            :class="
              isDraggingInspiration
                ? 'border-red-500 bg-red-950/20'
                : 'border-zinc-800 bg-zinc-950/50 hover:border-zinc-700 hover:bg-zinc-900/50'
            "
            @dragover.prevent="isDraggingInspiration = true"
            @dragleave.prevent="isDraggingInspiration = false"
            @drop.prevent="handleInspirationDrop"
            @click="inspirationInputRef?.click()"
          >
            <input
              ref="inspirationInputRef"
              type="file"
              accept="video/mp4,video/quicktime,.mp4,.mov"
              class="hidden"
              @change="handleInspirationSelect"
            />

            <template v-if="!inspirationFile">
              <Eye class="mb-2 h-6 w-6 text-red-500" />
              <p class="text-sm font-medium text-zinc-200">
                Glissez un TikTok de référence (ex: exemple.mp4) pour cloner son style
              </p>
              <p class="mt-1 text-xs text-zinc-500">
                Clonage automatique : format (9:16 ou 16:9), durée exacte, rythme des cuts francs et bande-son
              </p>
            </template>

            <div
              v-else
              class="flex w-full items-center justify-between rounded-lg bg-zinc-900/90 px-4 py-2.5"
              @click.stop
            >
              <div class="flex items-center gap-3 truncate">
                <Film class="h-5 w-5 shrink-0 text-red-400" />
                <div class="text-left truncate">
                  <p class="truncate text-sm font-semibold text-white">
                    {{ inspirationFile.name }}
                  </p>
                  <p class="text-xs text-zinc-400">
                    {{ formatFileSize(inspirationFile.size) }} • Format, Cuts & Audio clonés
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                @click="inspirationFile = null"
              >
                Retirer
              </Button>
            </div>
          </div>

          <label
            v-if="inspirationFile && !audioFile && !options.mute_output"
            class="mt-3 flex items-center justify-between rounded-xl border border-zinc-800/80 bg-zinc-950/60 px-3.5 py-2 cursor-pointer"
          >
            <span class="text-xs text-zinc-300">
              Extraire et utiliser automatiquement la musique de la vidéo d'inspiration
            </span>
            <input
              v-model="options.use_inspiration_audio"
              type="checkbox"
              class="h-4 w-4 accent-red-600 rounded"
            />
          </label>
        </Card>
      </div>

      <!-- COLONNE DROITE : CONFIGURATION FORMAT, RYTHME & COULEURS (5 colonnes) -->
      <div class="space-y-6 lg:col-span-5">
        <Card class="p-6 space-y-5">
          <div class="flex items-center gap-2.5 border-b border-zinc-800 pb-3">
            <Wand2 class="h-5 w-5 text-red-500" />
            <h2 class="text-lg font-bold text-white">
              Réglages du Montage Posing
            </h2>
          </div>

          <!-- Choix du Format : 9:16 ou 16:9 (Sans rognage ni recadrage) -->
          <div>
            <label class="block text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2">
              Format d'image (Sans rognage ni étirement)
            </label>
            <div class="grid grid-cols-2 gap-3">
              <button
                type="button"
                class="flex flex-col items-center justify-center rounded-xl border p-3 text-xs font-semibold transition-all"
                :class="
                  options.aspect_ratio === '9:16'
                    ? 'border-red-500 bg-red-950/30 text-white'
                    : 'border-zinc-800 bg-zinc-950/60 text-zinc-400 hover:border-zinc-700'
                "
                @click="options.aspect_ratio = '9:16'"
              >
                <span class="text-sm font-bold">9:16 Vertical</span>
                <span class="mt-0.5 text-[11px] opacity-75">TikTok / Reels (1080×1920)</span>
              </button>

              <button
                type="button"
                class="flex flex-col items-center justify-center rounded-xl border p-3 text-xs font-semibold transition-all"
                :class="
                  options.aspect_ratio === '16:9'
                    ? 'border-red-500 bg-red-950/30 text-white'
                    : 'border-zinc-800 bg-zinc-950/60 text-zinc-400 hover:border-zinc-700'
                "
                @click="options.aspect_ratio = '16:9'"
              >
                <span class="text-sm font-bold">16:9 Paysage</span>
                <span class="mt-0.5 text-[11px] opacity-75">Cinéma / YouTube (1920×1080)</span>
              </button>
            </div>

            <!-- Détection IA du bon sens & Interchange auto 9:16 <-> 16:9 (Sans rognage) -->
            <label
              class="mt-2.5 flex items-center justify-between rounded-xl border border-zinc-800/90 bg-zinc-950/60 px-3.5 py-2.5 cursor-pointer"
            >
              <div>
                <span class="text-xs font-bold text-zinc-200 block">
                  IA Sens Auto (Interchange 9:16 ↔ 16:9 sans rogner)
                </span>
                <span class="text-[11px] text-zinc-500">
                  Reconnaît le sens de la vidéo et bascule entre 9:16 et 16:9 sans rogner ni recadrer l'image
                </span>
              </div>
              <input
                v-model="options.auto_orient_and_frame"
                type="checkbox"
                class="h-4 w-4 accent-red-600 rounded"
              />
            </label>
          </div>

          <!-- Style de Montage : Pose Cut 1.0x (Style exemple.mp4) vs Speed Ramp -->
          <div>
            <label class="block text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2">
              Style de Coupe (Sans Transition)
            </label>
            <div class="grid grid-cols-2 gap-3">
              <button
                type="button"
                class="flex flex-col items-start rounded-xl border p-3 text-left text-xs transition-all"
                :class="
                  options.editing_mode === 'pose_cut'
                    ? 'border-red-500 bg-red-950/30 text-white'
                    : 'border-zinc-800 bg-zinc-950/60 text-zinc-400 hover:border-zinc-700'
                "
                @click="options.editing_mode = 'pose_cut'"
              >
                <span class="font-bold text-sm">Posing Cut 1.0x ★</span>
                <span class="mt-0.5 text-[11px] opacity-80">
                  Vitesse fluide naturelle, coupes franches sur les beats (comme exemple.mp4)
                </span>
              </button>

              <button
                type="button"
                class="flex flex-col items-start rounded-xl border p-3 text-left text-xs transition-all"
                :class="
                  options.editing_mode === 'speed_ramp'
                    ? 'border-red-500 bg-red-950/30 text-white'
                    : 'border-zinc-800 bg-zinc-950/60 text-zinc-400 hover:border-zinc-700'
                "
                @click="options.editing_mode = 'speed_ramp'"
              >
                <span class="font-bold text-sm">Speed Ramp Drop</span>
                <span class="mt-0.5 text-[11px] opacity-80">
                  Entrée rapide ({{ options.fast_speed_factor }}x) puis ralenti ({{ options.slow_speed_factor }}x)
                </span>
              </button>
            </div>
          </div>

          <!-- Durée cible de la vidéo finale -->
          <div>
            <div class="flex justify-between text-xs font-semibold">
              <span class="uppercase tracking-wider text-zinc-400">
                Durée cible du TikTok
              </span>
              <span class="text-red-400 font-mono">
                {{ options.target_duration.toFixed(1) }}s
              </span>
            </div>
            <input
              v-model.number="options.target_duration"
              type="range"
              min="10"
              max="45"
              step="0.5"
              class="mt-2 w-full accent-red-600 cursor-pointer"
            />
          </div>

          <!-- Option Plan d'intro (Mise en place sous la lumière) -->
          <label class="flex items-center justify-between rounded-xl border border-zinc-800/90 bg-zinc-950/60 px-3.5 py-2.5 cursor-pointer">
            <div>
              <span class="text-xs font-bold text-zinc-200 block">
                Inclure le plan d'intro (avant le 1er Beat Drop)
              </span>
              <span class="text-[11px] text-zinc-500">
                Mise en place sous la lumière de 0.0s à ~2.0s avant d'enchaîner les poses
              </span>
            </div>
            <input
              v-model="options.include_intro_shot"
              type="checkbox"
              class="h-4 w-4 accent-red-600 rounded"
            />
          </label>

          <!-- Fréquence des coupes (Sensibilité Beat Drops) -->
          <div>
            <div class="flex justify-between text-xs font-semibold">
              <span class="uppercase tracking-wider text-zinc-400">
                Rythme des Coupes (Beats Librosa)
              </span>
              <span class="font-mono text-zinc-200">
                {{ Math.round(options.drop_sensitivity * 100) }}%
              </span>
            </div>
            <input
              v-model.number="options.drop_sensitivity"
              type="range"
              min="0.2"
              max="0.9"
              step="0.05"
              class="mt-2 w-full accent-red-600"
            />
          </div>

          <!-- Choix du Look / Étalonnage FFmpeg -->
          <div>
            <label class="block text-xs font-semibold uppercase tracking-wider text-zinc-400">
              Rendu & Colorimétrie (FFmpeg)
            </label>
            <select
              v-model="options.lut_preset"
              class="mt-1.5 w-full rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-white focus:border-red-500 focus:outline-none"
            >
              <option value="natural_aesthetic">
                Naturel Net ★ (Fidèle au rush + Clarté musculaire — Style exemple.mp4)
              </option>
              <option value="dark_aesthetic">
                Dark Aesthetic Doux (Contraste subtil sans boucher les noirs)
              </option>
              <option value="obsidian_monochrome">
                Obsidian Monochrome (Noir & Blanc Bodybuilding)
              </option>
            </select>
          </div>

          <!-- Bouton d'action principal -->
          <Button
            size="lg"
            class="w-full"
            :disabled="!canLaunchPipeline"
            @click="launchAutomatedEdit"
          >
            <Loader2
              v-if="isUploading || isProcessing"
              class="h-5 w-5 animate-spin"
            />
            <Sparkles v-else class="h-5 w-5" />
            <span>
              {{
                isUploading
                  ? 'Upload des médias en cours...'
                  : isProcessing
                    ? 'Montage Posing en cours...'
                    : 'Générer le Montage TikTok Posing'
              }}
            </span>
          </Button>
        </Card>
      </div>
    </div>

    <!-- ===================================================================== -->
    <!-- PANNEAU DE PROGRESSION TEMPS-RÉEL & APERÇU DU RENDU FINAL             -->
    <!-- ===================================================================== -->
    <Card v-if="projectStatus" class="mt-8 p-6 space-y-6">
      <!-- En-tête de la barre de progression -->
      <div class="flex flex-col justify-between gap-2 sm:flex-row sm:items-center">
        <div>
          <div class="flex items-center gap-2.5">
            <Activity class="h-5 w-5 text-red-500 animate-pulse" />
            <h3 class="text-lg font-bold text-white">
              Progression du Pipeline — Projet #{{ projectStatus.project_id }}
            </h3>
          </div>
          <p class="mt-1 text-sm text-zinc-400">
            {{ projectStatus.current_step_detail }}
          </p>
        </div>
        <span class="font-mono text-2xl font-extrabold text-red-400">
          {{ progressPercent }}%
        </span>
      </div>

      <!-- Barre de progression Shadcn-vue -->
      <Progress :model-value="progressPercent" />

      <!-- Stepper visuel des 5 étapes -->
      <div class="grid grid-cols-1 gap-3 sm:grid-cols-5">
        <div
          v-for="step in PIPELINE_STEPS"
          :key="step.key"
          class="rounded-xl border p-3.5 transition-all"
          :class="{
            'border-emerald-700/70 bg-emerald-950/25 text-emerald-200':
              getStepState(step.key) === 'done',
            'border-red-500 bg-red-950/30 text-white shadow-lg shadow-red-950/40':
              getStepState(step.key) === 'active',
            'border-zinc-800/80 bg-zinc-950/40 text-zinc-500':
              getStepState(step.key) === 'pending',
          }"
        >
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold">{{ step.label }}</span>
            <CheckCircle2
              v-if="getStepState(step.key) === 'done'"
              class="h-4 w-4 text-emerald-400"
            />
            <Loader2
              v-else-if="getStepState(step.key) === 'active'"
              class="h-4 w-4 animate-spin text-red-400"
            />
          </div>
          <p class="mt-1 text-[11px] opacity-80">{{ step.sublabel }}</p>
        </div>
      </div>

      <!-- Résumé des Coupes & Poses sélectionnées -->
      <div
        v-if="projectStatus.selected_segments.length > 0"
        class="rounded-xl border border-zinc-800 bg-zinc-950/70 p-4"
      >
        <div class="mb-3 flex flex-wrap items-center justify-between gap-2">
          <span class="text-xs font-bold uppercase tracking-wider text-zinc-400">
            Timeline des Poses & Coupes Franches (Sans Transition)
          </span>
          <div class="flex flex-wrap items-center gap-2">
            <Badge
              v-if="projectStatus.effective_aspect_ratio"
              :variant="projectStatus.orientation_corrected ? 'success' : 'secondary'"
            >
              {{
                projectStatus.orientation_corrected
                  ? `IA Cadrage Auto : ${projectStatus.effective_aspect_ratio} (Redressé / Interchangé)`
                  : `Format : ${projectStatus.effective_aspect_ratio}`
              }}
            </Badge>
            <Badge v-if="projectStatus.inspiration_analysis" variant="secondary">
              Inspiration : {{ projectStatus.inspiration_analysis.detected_aspect_ratio }} •
              {{ projectStatus.inspiration_analysis.scene_cuts.length }} cuts clonés
            </Badge>
            <Badge v-if="projectStatus.audio_analysis" variant="default">
              {{
                projectStatus.audio_analysis.bpm > 0
                  ? `${projectStatus.audio_analysis.bpm} BPM`
                  : 'Sans Musique (Son à 0)'
              }}
              • {{ projectStatus.audio_analysis.effective_duration }}s
            </Badge>
          </div>
        </div>

        <div class="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
          <div
            v-for="(seg, idx) in projectStatus.selected_segments"
            :key="idx"
            class="rounded-lg border border-zinc-800/90 bg-zinc-900/60 p-2.5 text-xs"
          >
            <div class="flex items-center justify-between font-bold text-red-300">
              <span>Plan #{{ idx + 1 }} : {{ seg.pose_label }}</span>
              <span class="font-mono text-[11px] text-zinc-400">
                {{ seg.timeline_start.toFixed(1) }}s → {{ seg.timeline_end.toFixed(1) }}s
              </span>
            </div>
            <p class="mt-1 font-mono text-[11px] text-zinc-400">
              Rush source : {{ seg.start_time.toFixed(1) }}s – {{ seg.end_time.toFixed(1) }}s
              <span v-if="seg.rotation_correction" class="text-emerald-400">
                (Redressé {{ seg.rotation_correction }}°)
              </span>
            </p>
          </div>
        </div>
      </div>

      <!-- Aperçu Vidéo Direct & Téléchargements -->
      <div
        v-if="projectStatus.stage === 'completed' && finalVideoFullUrl"
        class="grid grid-cols-1 gap-6 rounded-xl border border-emerald-800/60 bg-emerald-950/15 p-5 lg:grid-cols-12 items-center"
      >
        <div class="lg:col-span-5 flex justify-center">
          <video
            :src="finalVideoFullUrl"
            controls
            playsinline
            class="max-h-96 rounded-xl border border-zinc-700 shadow-2xl bg-black"
          />
        </div>

        <div class="lg:col-span-7 space-y-4">
          <div>
            <Badge variant="success" class="mb-2">RENDU TERMINÉ</Badge>
            <h4 class="text-xl font-bold text-white">
              Votre TikTok Posing ({{ projectStatus.effective_aspect_ratio || options.aspect_ratio }}) est prêt !
            </h4>
            <p class="mt-1 text-sm text-zinc-300">
              Image non étirée (SAR 1:1), coupes franches synchronisées sur les
              beats sans transition et sélection multi-poses variée.
            </p>
          </div>

          <div class="flex flex-wrap items-center gap-3">
            <a
              :href="finalVideoFullUrl"
              download
              target="_blank"
            >
              <Button variant="default" size="lg">
                <Download class="h-4 w-4" />
                Télécharger la Vidéo (.MP4)
              </Button>
            </a>

            <a
              v-if="finalXmlFullUrl"
              :href="finalXmlFullUrl"
              download
              target="_blank"
            >
              <Button variant="secondary" size="lg">
                <FileCode2 class="h-4 w-4 text-red-400" />
                Exporter Timeline (.XML Premiere / Resolve)
              </Button>
            </a>
          </div>
        </div>
      </div>
    </Card>
  </div>
</template>
