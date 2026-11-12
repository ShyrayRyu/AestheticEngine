/**
 * Types et interfaces TypeScript alignés sur les schémas Pydantic v2 du backend FastAPI.
 */

export type PipelineStage =
  | 'uploaded'
  | 'analyzing_audio'
  | 'analyzing_vision'
  | 'speed_ramping'
  | 'color_grading'
  | 'exporting_edl'
  | 'completed'
  | 'failed'

export type AspectRatioOption = '9:16' | '16:9'
export type EditingModeOption = 'pose_cut' | 'speed_ramp'

export interface BeatDropEvent {
  timestamp: number
  intensity: number
  buildup_start: number
  sub_bass_energy: number
}

export interface AudioAnalysisResult {
  bpm: number
  duration: number
  effective_duration: number
  beat_timestamps: number[]
  beat_drops: BeatDropEvent[]
}

export interface FrameEffortScore {
  video_filename: string
  timestamp: number
  effort_score: number
  exercise_detected: string
  is_peak_contraction: boolean
  reasoning: string
}

export interface SelectedVideoSegment {
  source_video: string
  start_time: number
  end_time: number
  peak_time_in_source: number
  timeline_start: number
  timeline_end: number
  pose_label: string
  max_effort_score: number
  target_beat_drop: number | null
  speed_fast_factor: number
  speed_slow_factor: number
  rotation_correction?: number
  upright_aspect_ratio?: AspectRatioOption
  focus_x?: number
  focus_y?: number
}

export interface InspirationAnalysisResult {
  filename: string
  duration: number
  detected_aspect_ratio: AspectRatioOption
  scene_cuts: number[]
  audio_extracted: boolean
}

export interface ProcessProjectRequest {
  aspect_ratio: AspectRatioOption
  auto_orient_and_frame: boolean
  editing_mode: EditingModeOption
  target_duration: number
  include_intro_shot: boolean
  use_inspiration_audio: boolean
  mute_output: boolean
  min_effort_score: number
  ollama_model: string
  fast_speed_factor: number
  slow_speed_factor: number
  lut_preset: string
  drop_sensitivity: number
}

export interface UploadResponse {
  project_id: string
  message: string
  video_files: string[]
  audio_file: string | null
  inspiration_file: string | null
}

export interface ProjectStatusResponse {
  project_id: string
  stage: PipelineStage
  progress_percent: number
  current_step_detail: string
  video_files: string[]
  audio_file: string | null
  inspiration_file: string | null
  inspiration_analysis: InspirationAnalysisResult | null
  effective_aspect_ratio?: AspectRatioOption
  orientation_corrected?: boolean
  audio_analysis: AudioAnalysisResult | null
  frame_scores: FrameEffortScore[]
  selected_segments: SelectedVideoSegment[]
  output_video_url: string | null
  output_xml_url: OptionalString
  error_message: string | null
  logs: string[]
}

type OptionalString = string | null
