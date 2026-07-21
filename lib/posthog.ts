import PostHog from "posthog-react-native"
import Constants from "expo-constants"

const apiKey Constants.expoConfig?.extra?.posthogProjectToken as 
| string
| undefined
const host = Constants.expoConfig?.extra?.posthog as string | undefined
const isPostHogConfigured =
!!apiKey && apiKey !== 