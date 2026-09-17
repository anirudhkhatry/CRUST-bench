// Stand-in for `react-native-safe-area-context`, which cannot be installed alongside
// react-native-web without npm re-resolving the whole tree and dropping
// @testing-library/dom (measured: `--legacy-peer-deps` removes 10 packages and every
// React fixture stops loading). These fixtures mock `react-native` in their own holdouts,
// so the safe-area API only has to RESOLVE and return plausible insets -- nothing under
// test depends on real device metrics.
import { createElement } from 'react';

export const initialWindowMetrics = {
  frame: { x: 0, y: 0, width: 375, height: 812 },
  insets: { top: 44, left: 0, right: 0, bottom: 34 },
};
const INSETS = { top: 0, left: 0, right: 0, bottom: 0 };
export const useSafeAreaInsets = () => INSETS;
export const useSafeAreaFrame = () => initialWindowMetrics.frame;
export const SafeAreaProvider = ({ children }) => children ?? null;
export const SafeAreaConsumer = ({ children }) => children?.(INSETS) ?? null;
export const SafeAreaInsetsContext = { Provider: SafeAreaProvider, Consumer: SafeAreaConsumer };
export const SafeAreaView = ({ children, ...props }) =>
  createElement('div', props, children);
export const withSafeAreaInsets = (C) => (props) =>
  createElement(C, { ...props, insets: INSETS });
export default { SafeAreaProvider, SafeAreaView, useSafeAreaInsets };
