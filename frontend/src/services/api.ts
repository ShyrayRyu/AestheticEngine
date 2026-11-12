/**
 * Service client HTTP & WebSocket communiquant avec le backend FastAPI d'AestheticEngine.
 */

import type {
  ProcessProjectRequest,
  ProjectStatusResponse,
  UploadResponse,
} from '@/types/project'

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
export const WS_BASE_URL =
  import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000'

/**
 * Envoie les rushs vidéo (.mp4), la piste audio optionnelle (.mp3/.wav) et la vidéo
 * d'inspiration optionnelle (.mp4) au backend via Multipart FormData.
 */
export async function uploadProjectMedia(
  videos: File[],
  audio?: File | null,
  inspiration?: File | null
): Promise<UploadResponse> {
  const formData = new FormData()
  for (const video of videos) {
    formData.append('videos', video)
  }
  if (audio) {
    formData.append('audio', audio)
  }
  if (inspiration) {
    formData.append('inspiration', inspiration)
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/projects/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const errBody = await response.json().catch(() => ({}))
    throw new Error(
      errBody.detail || `Erreur d'upload HTTP ${response.status}`
    )
  }

  return response.json()
}

/**
 * Déclenche le pipeline de traitement (Librosa + Ollama LLaVA + MoviePy + FFmpeg LUT + XML EDL).
 */
export async function startProjectProcessing(
  projectId: string,
  options: ProcessProjectRequest
): Promise<ProjectStatusResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/projects/${projectId}/process`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(options),
    }
  )

  if (!response.ok) {
    const errBody = await response.json().catch(() => ({}))
    throw new Error(
      errBody.detail || `Erreur lors du lancement HTTP ${response.status}`
    )
  }

  return response.json()
}

/**
 * Récupère l'état d'avancement d'un projet par polling HTTP.
 */
export async function fetchProjectStatus(
  projectId: string
): Promise<ProjectStatusResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/projects/${projectId}/status`
  )
  if (!response.ok) {
    throw new Error(`Impossible de récupérer le statut (${response.status})`)
  }
  return response.json()
}
